using System;
using System.Collections.Generic;
using System.Linq;

namespace FarmerLShapedNoILOG
{
    class Program
    {
        // Original farmer problem variables in order: [x1, x2, x3, y1, y2, w1, w2, w3, w4, theta]
        // We add 'theta' for L-shaped method

        // Master constraints: user picks which constraints go in master.
        // The rest + scenario yield logic goes to subproblem.

        // Let's store the original problem constraints in a data structure:
        static double[,] originalLHS = new double[5, 10];  // 5 constraints, 10 vars
        static double[] originalRHS = new double[5];
        static string[] consLabels = {"C1","C2","C3","C4","C5"};
        static Dictionary<string,int> consIndexMap = new Dictionary<string,int>{{"C1",0},{"C2",1},{"C3",2},{"C4",3},{"C5",4}};
        static List<string> allVars = new List<string>{"x_1","x_2","x_3","y_1","y_2","w_1","w_2","w_3","w_4","theta"};

        // Objective coefficients (original):
        static Dictionary<string,double> varObj = new Dictionary<string,double>{
            {"x_1",150},{"x_2",230},{"x_3",260},
            {"y_1",238},{"y_2",210},
            {"w_1",-170},{"w_2",-150},{"w_3",-36},{"w_4",-10},
            {"theta",0.0}
        };

        // For subproblem solutions, we need subproblem objective only for {y,w}:
        // But here we define subObj for {y1,y2,w1,w2,w3,w4} (the subproblem variables).
        // The code will unify them from varObj.

        static void Main(string[] args)
        {
            // (1) Print Original Problem
            PrintOriginalProblem();

            // Fill the original constraints LHS, RHS
            // C1: x1+x2+x3 ≤ 500
            originalLHS[0,0]=1; originalLHS[0,1]=1; originalLHS[0,2]=1; originalRHS[0]=500;
            // C2: 2.5x1 + y1 - w1≥200 => -2.5x1 -y1 +w1 ≤ -200
            originalLHS[1,0]=-2.5; originalLHS[1,3]=-1; originalLHS[1,5]=1; originalRHS[1]=-200;
            // C3: 3x2 + y2 - w2≥240 => -3x2 -y2 + w2 ≤ -240
            originalLHS[2,1]=-3; originalLHS[2,4]=-1; originalLHS[2,6]=1; originalRHS[2]=-240;
            // C4: w3+w4 ≤20x3 => w3+w4 -20x3≤0
            originalLHS[3,7]=1; originalLHS[3,8]=1; originalLHS[3,2]=-20; originalRHS[3]=0;
            // C5: w3 ≤6000
            originalLHS[4,7]=1; originalRHS[4]=6000;

            // (2) Let user pick constraints for the master
            Console.WriteLine("\nEnter constraints for master (e.g. C1,C2...):");
            string inputCons=Console.ReadLine();
            string[] chosenConsStr = inputCons.Split(',').Select(s=>s.Trim().ToUpper()).Where(s=>s!="").ToArray();
            HashSet<int> chosenConsIdx=new HashSet<int>();
            foreach(var c in chosenConsStr)
            {
                if(!consIndexMap.ContainsKey(c))
                {
                    Console.WriteLine($"Constraint {c} not found. Stop.");
                    return;
                }
                chosenConsIdx.Add(consIndexMap[c]);
            }

            // (3) Let user define number of scenarios
            Console.WriteLine("Enter number of scenarios:");
            int numScenarios;
            while(!int.TryParse(Console.ReadLine(), out numScenarios) || numScenarios<=0)
            {
                Console.WriteLine("Invalid input. Re-enter number of scenarios:");
            }

            // Build Master Problem
            (double[,] masterLHS, double[] masterRHS, double[] masterCoeffs, List<string> masterVarNames) =
                BuildMasterProblemData(chosenConsIdx);

            // We'll do a primal simplex approach for the master problem, so let's keep it in memory.
            // master dimension:
            int masterCons = masterRHS.Length;
            int masterVarsCount = masterCoeffs.Length;

            // Subproblem constraints = the constraints not chosen
            List<int> subConsIndices=new List<int>();
            for(int i=0;i<5;i++)
                if(!chosenConsIdx.Contains(i)) subConsIndices.Add(i);

            // Subproblem variables = the original minus the master var set (and "theta")
            // We find the name "theta" or slack not in subproblem
            HashSet<string> masterVarSet = new HashSet<string>(masterVarNames);
            List<int> subVarIndices=new List<int>();
            for(int idx=0; idx<allVars.Count; idx++)
            {
                string v=allVars[idx];
                if(!masterVarSet.Contains(v) && !v.StartsWith("slack_") && v!="theta")
                {
                    subVarIndices.Add(idx);
                }
            }

            // Check complete recourse: If user picks any constraint that has y_ or w_ => not complete recourse
            bool isCompleteRecourse=true;
            foreach(int ci in chosenConsIdx)
            {
                // Check if that constraint includes any y_ or w_
                for(int j=0;j<10;j++)
                {
                    if(Math.Abs(originalLHS[ci,j])>1e-15)
                    {
                        string varName=allVars[j];
                        if(varName.StartsWith("y_")||varName.StartsWith("w_"))
                            isCompleteRecourse=false;
                    }
                }
            }

            // (4) Build scenario yields from 80% ~120%
            double[] scenarioMultipliers=ComputeScenarioMultipliers(numScenarios);

            // L-shaped iteration
            double UB=double.PositiveInfinity;
            double LB=double.NegativeInfinity;
            double prevTheta= double.NegativeInfinity;
            bool optimalFound=false;
            int maxIter=50;

            // We'll store the master tableau for primal simplex:
            (double[,] masterTableau, double[] masterRHSVec, double[] masterObjVec)= BuildMasterTableau(masterLHS, masterRHS, masterCoeffs);

            int idxTheta=-1; // identify "theta" in masterVarNames
            for(int i=0;i<masterVarNames.Count;i++)
            {
                if(masterVarNames[i]=="theta")
                {
                    idxTheta=i;break;
                }
            }
            if(idxTheta<0)
            {
                Console.WriteLine("No 'theta' in master? That breaks L-shaped logic. Stop.");
                return;
            }

            double[] lastMasterSolution=null;
            for(int iteration=1; iteration<=maxIter; iteration++)
            {
                Console.WriteLine($"\n--- L-Shaped Iteration {iteration} ---");
                var masterRes=PrimalSimplexSolveFull(masterTableau, masterRHSVec, masterObjVec);
                if(!masterRes.feasible)
                {
                    Console.WriteLine("Master infeasible. Stop iteration.");
                    break;
                }
                double[] mSol= masterRes.solution;
                double objVal= masterRes.objVal;
                // Print master solution
                Console.WriteLine("Master solution:");
                for(int v=0;v<masterVarNames.Count;v++)
                {
                    Console.WriteLine($"{masterVarNames[v]}={mSol[v]:F2}");
                }
                Console.WriteLine($"MasterObj={objVal:F2}");

                UB=Math.Min(UB,objVal);
                double thetaVal=mSol[idxTheta];

                // Solve subproblems
                bool anyFeasCut=false;
                double totalScenarioCost=0.0;
                for(int s=0;s<numScenarios;s++)
                {
                    double multiplier= scenarioMultipliers[s];
                    double wheatYield=2.5*multiplier;
                    double cornYield=3.0*multiplier;
                    double sugarYield=20.0*multiplier;

                    var sp= SolveSubproblemScenario(mSol,masterVarNames, subConsIndices, subVarIndices, 
                             wheatYield, cornYield, sugarYield, isCompleteRecourse);
                    if(!sp.feasible && !isCompleteRecourse)
                    {
                        anyFeasCut=true;
                        AddFeasibilityCutSimple(ref masterTableau,ref masterRHSVec,ref masterObjVec,masterVarNames);
                        Console.WriteLine("Feasibility cut added!");
                        break;
                    }
                    else if(!sp.feasible && isCompleteRecourse)
                    {
                        Console.WriteLine("Infeasible scenario under complete recourse?? Contradiction. Stop.");
                        anyFeasCut=true;
                        break;
                    }
                    else
                    {
                        totalScenarioCost+= sp.cost*(1.0/numScenarios);
                    }
                }
                if(anyFeasCut) continue;

                // All scenario feasible
                if(!double.IsNegativeInfinity(prevTheta))
                {
                    double diff=Math.Abs(thetaVal - prevTheta);
                    if(diff<1e-6)
                    {
                        Console.WriteLine("Theta converged => stop iteration.");
                        optimalFound=true;
                        break;
                    }
                }
                prevTheta=thetaVal;

                double w_v=totalScenarioCost;
                if(thetaVal>=w_v-1e-6)
                {
                    LB=Math.Max(LB,thetaVal);
                    if(Math.Abs(UB-LB)<1e-6)
                    {
                        Console.WriteLine("Converged: UB ~ LB");
                        optimalFound=true;
                        break;
                    }
                }
                else
                {
                    // Add optimality cut: theta >= w_v => (theta - w_v >=0)
                    // We'll do that with primal approach by adding a row
                    AddOptimalityCut(ref masterTableau,ref masterRHSVec,ref masterObjVec, idxTheta, w_v, masterVarNames.Count);
                    Console.WriteLine("Optimality cut added!");
                }
            }

            if(optimalFound)
            {
                // final solution
                var final=PrimalSimplexSolveFull(masterTableau,masterRHSVec,masterObjVec);
                if(final.feasible)
                {
                    Console.WriteLine("\n--- Final Master Solution ---");
                    double[] sol=final.solution;
                    double fObj=final.objVal;
                    for(int i=0;i<masterVarNames.Count;i++)
                    {
                        if(!masterVarNames[i].StartsWith("slack_"))
                            Console.WriteLine($"{masterVarNames[i]}={sol[i]:F2}");
                    }
                    Console.WriteLine($"Objective={fObj:F2}");
                }
            }
            else
            {
                Console.WriteLine("Max iteration or no converge reached.");
            }

            // Print Advice
            Console.WriteLine("\n--- Advice ---");
            Console.WriteLine("1) This code uses an internal primal simplex approach for both master and subproblems.");
            Console.WriteLine("2) If user-chosen constraints do not include any y_/w_, the problem is complete recourse => no feasibility cuts needed.");
            Console.WriteLine("3) For accurate L-shaped cuts, we typically gather subproblem dual multipliers. Here we used a direct subproblem cost (w_v).");
            Console.WriteLine("4) You can refine the logic by forming feasibility/optimality cuts from the subproblem dual solutions for a robust solution.");
        }

        // Build Master Problem Data:
        static (double[,] lhs, double[] rhs, double[] coeffs, List<string> varNames)
        BuildMasterProblemData(HashSet<int> chosenConsIdx)
        {
            // gather all vars from chosen constraints + "theta"
            int thetaIndex= allVars.IndexOf("theta");
            HashSet<int> masterVarIndices=new HashSet<int>();
            foreach(int ci in chosenConsIdx)
            {
                for(int j=0;j<10;j++)
                {
                    if(Math.Abs(originalLHS[ci,j])>1e-15)
                        masterVarIndices.Add(j);
                }
            }
            masterVarIndices.Add(thetaIndex);

            List<int> mList=new List<int>(masterVarIndices);
            mList.Sort();
            int masterCons=chosenConsIdx.Count;
            int slack= masterCons;
            int totalVars= mList.Count + slack;

            double[,] masterLHS=new double[masterCons, totalVars];
            double[] masterRHS=new double[masterCons];
            List<string> masterVarNames=new List<string>();
            for(int j=0;j<mList.Count;j++)
            {
                masterVarNames.Add(allVars[mList[j]]);
            }
            for(int i=0;i<masterCons;i++)
            {
                masterVarNames.Add($"slack_{i+1}");
            }

            int row=0;
            foreach(int ci in chosenConsIdx)
            {
                for(int j=0;j<mList.Count;j++)
                {
                    masterLHS[row,j]= originalLHS[ci,mList[j]];
                }
                masterLHS[row,mList.Count+row]=1.0;
                masterRHS[row]=originalRHS[ci];
                row++;
            }

            // build master objective
            double[] masterCoeffs=new double[totalVars];
            for(int j=0;j<mList.Count;j++)
            {
                string v=allVars[mList[j]];
                masterCoeffs[j]= varObj[v];
            }
            for(int i=0;i<masterCons;i++)
            {
                masterCoeffs[mList.Count+i]=0.0;
            }

            return (masterLHS, masterRHS, masterCoeffs, masterVarNames);
        }

        // Build initial tableau for primal simplex
        static (double[,] lhs,double[] rhs,double[] obj) BuildMasterTableau(double[,] masterLHS, double[] masterRHS, double[] masterCoeffs)
        {
            int m=masterRHS.Length;  // #constraints
            int tvars=masterCoeffs.Length; //#vars
            // We'll transform Ax ≤ b into a tableau form
            // The last m columns will be the slack variables, presumably an identity basis.
            double[,] lhs=new double[m,tvars];
            double[] rhs=new double[m];
            double[] obj=new double[tvars];
            for(int i=0;i<m;i++)
            {
                for(int j=0;j<tvars;j++)
                {
                    lhs[i,j]=masterLHS[i,j];
                }
                rhs[i]=masterRHS[i];
            }
            for(int j=0;j<tvars;j++)
                obj[j]=masterCoeffs[j];
            return (lhs,rhs,obj);
        }

        // Solve subproblem scenario with internal approach
        // For demonstration, we do a direct approach:
        // subproblem constraints = subConsIndices with yield variations substituted
        // subproblem objective from varObj for subVarIndices
        public struct SubproblemResult {public bool feasible;public double cost; }
        static SubproblemResult SolveSubproblemScenario(double[] masterSolVal,List<string> masterVarNames,
            List<int> subConsIndices,List<int> subVarIndices, double wheatYield, double cornYield, double sugarYield,
            bool isCompleteRecourse)
        {
            // Build subproblem tableau
            // subproblem constraints => subConsIndices
            // subproblem variables => subVarIndices
            // objective => sum(varObj[vName]*subVar)
            // For demonstration: we'll treat constraints as ≤ form, add slack => BFS
            // Then do primal simplex.

            // Build LHS,RHS for subproblem
            int subM=subConsIndices.Count;
            int subVarCount= subVarIndices.Count;
            double[,] subLHS=new double[subM, subVarCount + subM];
            double[] subRHS=new double[subM];
            double[] subObj=new double[subVarCount + subM];

            for(int j=0;j<subVarCount;j++)
            {
                string v= allVars[subVarIndices[j]];
                subObj[j]= varObj[v];
            }
            for(int i=0;i<subM;i++) subObj[subVarCount+i]=0.0;

            for(int i=0;i<subM;i++)
            {
                int ci=subConsIndices[i];
                double[] cRow=new double[10];
                for(int kk=0;kk<10;kk++) cRow[kk]=originalLHS[ci,kk];
                double cRHS= originalRHS[ci];

                // Adjust yields:
                if(ci==1) cRow[0]=-wheatYield; //C2
                if(ci==2) cRow[1]=-cornYield;  //C3
                if(ci==3) cRow[2]=-sugarYield; //C4

                // Sub out master variables
                double hVal=cRHS;
                for(int mv=0; mv<masterVarNames.Count; mv++)
                {
                    string varM= masterVarNames[mv];
                    int globalIdx= allVars.IndexOf(varM);
                    if(globalIdx>=0 && Math.Abs(cRow[globalIdx])>1e-15)
                    {
                        hVal-= cRow[globalIdx]*masterSolVal[mv];
                        cRow[globalIdx]=0.0;
                    }
                }

                // Fill subLHS row
                for(int j=0;j<subVarCount;j++)
                {
                    int globalID=subVarIndices[j];
                    subLHS[i,j]= cRow[globalID];
                }
                subLHS[i, subVarCount+i]=1.0; // slack
                subRHS[i]=hVal;
            }

            // Now primal simplex
            var spTableau= BuildSubproblemTableau(subLHS, subRHS, subObj, subM, subVarCount);
            var spRes=PrimalSimplexSolveFull(spTableau.lhs, spTableau.rhs, spTableau.obj);
            SubproblemResult result = new SubproblemResult();
            result.feasible= spRes.feasible;
            if(!spRes.feasible) 
            {
                result.cost=Double.PositiveInfinity;
                return result;
            }
            result.cost= spRes.objVal;
            return result;
        }

        static (double[,] lhs,double[] rhs,double[] obj) BuildSubproblemTableau(double[,] subLHS,double[] subRHS,double[] subObj,int subM,int subVarCount)
        {
            // subproblem total vars = subVarCount + subM
            // We'll transform them into a tableau
            int totalVars=subVarCount+subM;
            double[,] lhs=new double[subM,totalVars];
            double[] rhs=new double[subM];
            double[] obj=new double[totalVars];
            for(int i=0;i<subM;i++)
            {
                for(int j=0;j<totalVars;j++)
                    lhs[i,j]=subLHS[i,j];
                rhs[i]=subRHS[i];
            }
            for(int j=0;j<totalVars;j++)
                obj[j]=subObj[j];

            return (lhs,rhs,obj);
        }

        // Primal Simplex result
        public struct SimplexResult
        {
            public bool feasible;
            public double[] solution;
            public double objVal;
        }

        // Full primal simplex solver 
        static SimplexResult PrimalSimplexSolveFull(double[,] lhs,double[] rhs,double[] obj)
        {
            // We assume Ax <= b, all vars≥0, slack provides BFS
            // We'll do standard primal simplex approach
            // The # of constraints = m
            // The # of total vars = n
            int m=rhs.Length;
            int n=obj.Length;
            // Make a local copy so we don't mutate the original arrays
            double[,] A=new double[m,n];
            double[] B=new double[m];
            double[] c=new double[n];
            for(int i=0;i<m;i++)
            {
                B[i]=rhs[i];
                for(int j=0;j<n;j++)
                    A[i,j]=lhs[i,j];
            }
            for(int j=0;j<n;j++)
                c[j]=obj[j];

            // Basic variables = last m columns (the slacks)
            int[] basicVar = new int[m];
            for(int i=0;i<m;i++)
            {
                basicVar[i]=n-m+i;
            }

            // iteration
            while(true)
            {
                // compute dual pi from basicVar
                double[] pi=new double[m];
                for(int i=0;i<m;i++)
                {
                    pi[i]= c[basicVar[i]];
                }

                // compute reduced cost
                double maxRC=0.0; int entering=-1;
                double[] redCost=new double[n];
                for(int j=0;j<n;j++)
                {
                    redCost[j]= c[j];
                    for(int i=0;i<m;i++)
                        redCost[j]-= pi[i]*A[i,j];
                    if(redCost[j]>maxRC)
                    {
                        maxRC=redCost[j];
                        entering=j;
                    }
                }
                if(entering==-1) 
                {
                    // optimal
                    break;
                }

                // ratio test
                double minRatio=Double.MaxValue;
                int leaving=-1;
                for(int i=0;i<m;i++)
                {
                    if(A[i,entering]>1e-15)
                    {
                        double ratio= B[i]/A[i,entering];
                        if(ratio<minRatio)
                        {
                            minRatio=ratio;
                            leaving=i;
                        }
                    }
                }
                if(leaving==-1)
                {
                    // unbounded
                    SimplexResult res=new SimplexResult();
                    res.feasible=false;
                    res.solution=null;
                    res.objVal=Double.PositiveInfinity;
                    return res;
                }

                // pivot
                double pivot=A[leaving,entering];
                for(int j=0;j<n;j++)
                {
                    A[leaving,j]/=pivot;
                }
                B[leaving]/=pivot;
                double objFactor=c[entering];
                for(int j=0;j<n;j++)
                {
                    c[j]-= objFactor*A[leaving,j];
                }
                for(int i=0;i<m;i++)
                {
                    if(i!=leaving)
                    {
                        double factor=A[i,entering];
                        for(int j=0;j<n;j++)
                        {
                            A[i,j]-= factor*A[leaving,j];
                        }
                        B[i]-= factor*B[leaving];
                    }
                }
                basicVar[leaving]= entering;
            }

            // build solution
            double[] solution=new double[n];
            for(int i=0;i<m;i++)
            {
                solution[basicVar[i]]=B[i];
            }
            double objVal=0.0;
            for(int j=0;j<n;j++)
            {
                objVal+= solution[j]*obj[j];
            }
            SimplexResult finalRes=new SimplexResult();
            finalRes.feasible=true;
            finalRes.solution=solution;
            finalRes.objVal=objVal;
            return finalRes;
        }

        static void AddFeasibilityCutSimple(ref double[,] masterLHS,ref double[] masterRHS, ref double[] masterObj, List<string> masterVarNames)
        {
            // Add a simple feasibility cut: -x_1 - x_2 -x_3≥ -100 if they exist
            int oldCons=masterRHS.Length;
            int oldVars=masterObj.Length;
            double[,] newLHS=new double[oldCons+1, oldVars];
            double[] newRHS=new double[oldCons+1];
            for(int i=0;i<oldCons;i++)
            {
                for(int j=0;j<oldVars;j++)
                {
                    newLHS[i,j]= masterLHS[i,j];
                }
                newRHS[i]= masterRHS[i];
            }
            // find x_1,x_2,x_3 in masterVarNames
            int x1=masterVarNames.IndexOf("x_1");
            int x2=masterVarNames.IndexOf("x_2");
            int x3=masterVarNames.IndexOf("x_3");
            for(int j=0;j<oldVars;j++)
            {
                newLHS[oldCons,j]=0.0;
            }
            if(x1>=0) newLHS[oldCons,x1]=-1.0;
            if(x2>=0) newLHS[oldCons,x2]=-1.0;
            if(x3>=0) newLHS[oldCons,x3]=-1.0;
            newRHS[oldCons]=-100.0;

            masterLHS=newLHS; masterRHS=newRHS;
        }

        static void AddOptimalityCut(ref double[,] masterLHS,ref double[] masterRHS,ref double[] masterObj, int idxTheta, double w_v, int totalVars)
        {
            // Add row: theta≥ w_v => row: theta - w_v≥0
            int oldCons= masterRHS.Length;
            double[,] newLHS=new double[oldCons+1, totalVars];
            double[] newRHS=new double[oldCons+1];
            for(int i=0;i<oldCons;i++)
            {
                for(int j=0;j<totalVars;j++)
                    newLHS[i,j]=masterLHS[i,j];
                newRHS[i]=masterRHS[i];
            }
            for(int j=0;j<totalVars;j++)
                newLHS[oldCons,j]=0.0;
            newLHS[oldCons, idxTheta]=1.0;
            newRHS[oldCons]= w_v;

            masterLHS=newLHS; masterRHS=newRHS;
        }

        static void PrintOriginalProblem()
        {
            Console.WriteLine("--- Original Farmer's Problem (No ILOG)---");
            Console.WriteLine("Objective: min 150x1 +230x2 +260x3 +238y1 -170w1 +210y2 -150w2 -36w3 -10w4");
            Console.WriteLine("Constraints:");
            Console.WriteLine("C1: x1+x2+x3 ≤500");
            Console.WriteLine("C2: 2.5x1 + y1 - w1 ≥200 => -2.5x1 - y1 + w1 ≤ -200");
            Console.WriteLine("C3: 3x2 + y2 - w2 ≥240 => -3x2 - y2 + w2 ≤ -240");
            Console.WriteLine("C4: w3+w4 ≤20x3 => w3+w4 -20x3≤0");
            Console.WriteLine("C5: w3 ≤6000");
        }

        static void PrintConstraint(double[,] LHS, int row, List<string> varNames,double rhs)
        {
            bool first=true;
            for(int j=0;j<varNames.Count;j++)
            {
                double coeff=LHS[row,j];
                if(Math.Abs(coeff)>1e-15)
                {
                    if(!first && coeff>0) Console.Write("+ ");
                    Console.Write($"{coeff:F2}*{varNames[j]} ");
                    first=false;
                }
            }
            Console.WriteLine($" ≤ {rhs:F2}");
        }

        static double[] ComputeScenarioMultipliers(int numScenarios)
        {
            double[] arr=new double[numScenarios];
            if(numScenarios==1)
            {
                arr[0]=1.0; return arr;
            }
            double step=0.4/(numScenarios-1); // from 0.8 to 1.2 => range0.4
            for(int i=0;i<numScenarios;i++)
            {
                arr[i]=0.8 + i*step;
            }
            return arr;
        }
    }
}
