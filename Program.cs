using System;
using System.Collections.Generic;
using System.Linq;

namespace FarmerLShapedFinal
{
    class Program
    {
        /*
         * Farmer Problem Data:
         *  - 500 acres total
         *  - Crops: Wheat, Corn, Sugarbeets
         * 
         * Yields & Costs:
         *   Wheat: yield 2.5 T/acre, plant cost 150 $/acre
         *          selling price 170 $/T, purchase price 238 $/T, min demand 200 T
         *   Corn:  yield 3   T/acre, plant cost 230 $/acre
         *          selling price 150 $/T, purchase price 210 $/T, min demand 240 T
         *   Sugar: yield 20  T/acre, plant cost 260 $/acre
         *          selling price: 36 $/T for up to 6000 T, 10 $/T above 6000 T
         *          no min demand, no purchase
         *
         * Variables (x_1,x_2,x_3) => farmland for wheat,corn,sugar
         * Second-stage: y_1,y_2 => shortfalls of wheat,corn (buy)
         *               w_1,w_2 => surplus of wheat,corn (sell)
         *               w_3,w_4 => sugarbeet selling (2-tier)
         * 
         * Objective: min [150x1 +230x2 +260x3] + [238y1 +210y2 -170w1 -150w2 -36w3 -10w4]
         * 
         * Constraints:
         *   C1: x1 + x2 + x3 <= 500
         *   C2: 2.5x1 + y1 - w1 >= 200  => -2.5x1 - y1 + w1 <= -200
         *   C3: 3x2 + y2 - w2 >= 240    => -3x2 - y2 + w2 <= -240
         *   C4: w3 + w4 <= 20 x3        => w3 + w4 -20x3 <= 0
         *   C5: w3 <= 6000              => w3 <= 6000
         *
         * Non-negativity: x1,x2,x3 >=0, y1,y2 >=0, w1,w2,w3,w4 >=0
         *
         * We do L-shaped decomposition:
         *   - The user picks which constraints go in MASTER. 
         *   - The rest + constraints + scenario yield logic => SUBPROBLEM.
         *   - The yield can vary from 80% to 120%.
         *
         * This code fully enforces non-negativity in subproblem (and master), 
         * 
         */

        // 5 constraints (C1..C5) in "≤" form, 10 variables: [x1,x2,x3,y1,y2,w1,w2,w3,w4,theta]
        static double[,] originalLHS = new double[5,10];  // 5 constraints, 10 vars
        static double[] originalRHS = new double[5];

        static string[] consLabels = { "C1", "C2", "C3", "C4", "C5" };
        static Dictionary<string,int> consIndexMap = new Dictionary<string,int> {
            {"C1",0},{"C2",1},{"C3",2},{"C4",3},{"C5",4}
        };

        // The 10 variables
        static List<string> allVars = new List<string> {
            "x_1","x_2","x_3",
            "y_1","y_2",
            "w_1","w_2","w_3","w_4",
            "theta"
        };

        // Objective Coeffs
        static Dictionary<string,double> varObj = new Dictionary<string,double> {
            {"x_1",150}, {"x_2",230}, {"x_3",260},
            {"y_1",238}, {"y_2",210},
            {"w_1",-170},{"w_2",-150},{"w_3",-36},{"w_4",-10},
            {"theta",1.0}
        };

        static void Main(string[] args)
        {
            Console.WriteLine("=== Farmer Problem with L-Shaped Decomposition (with Non-negativity) ===");

            // Fill original constraints
            // C1: x1 + x2 + x3 <=500
            originalLHS[0,0]=1; originalLHS[0,1]=1; originalLHS[0,2]=1; originalRHS[0]=500;

            // C2: 2.5x1 + y1 - w1 >=200 => -2.5x1 -y1 +w1 <=-200
            originalLHS[1,0]= -2.5; originalLHS[1,3]= -1; originalLHS[1,5]= 1; originalRHS[1]= -200;

            // C3: 3x2 + y2 - w2 >=240 => -3x2 -y2 + w2 <=-240
            originalLHS[2,1]= -3; originalLHS[2,4]= -1; originalLHS[2,6]= 1; originalRHS[2]= -240;

            // C4: w3 + w4 <=20x3 => w3 +w4 -20x3 <=0
            originalLHS[3,7]=1; originalLHS[3,8]=1; originalLHS[3,2]= -20; originalRHS[3]=0;

            // C5: w3 <=6000
            originalLHS[4,7]=1; originalRHS[4]=6000;

            PrintOriginalProblem();

            // (1) Let user pick constraints for MASTER
            Console.WriteLine("\nEnter constraints for MASTER (e.g. C1,C2...):");
            string line= Console.ReadLine();
            string[] chosenConsStr = line.Split(',')
                .Select(s=>s.Trim().ToUpper())
                .Where(s=>s!="").ToArray();

            HashSet<int> chosenConsIdx = new HashSet<int>();
            foreach(var c in chosenConsStr)
            {
                if(!consIndexMap.ContainsKey(c))
                {
                    Console.WriteLine($"Constraint {c} not found. Stop.");
                    return;
                }
                chosenConsIdx.Add(consIndexMap[c]);
            }

            // (2) number of scenarios
            Console.WriteLine("Enter number of scenarios:");
            int numScenarios;
            while(!int.TryParse(Console.ReadLine(), out numScenarios) || numScenarios<=0)
                Console.WriteLine("Invalid, re-enter #scenarios:");

            // Build Master Problem
            var masterData = BuildMasterProblemData(chosenConsIdx);
            double[,] masterLHS = masterData.lhs;
            double[] masterRHS = masterData.rhs;
            double[] masterCoeffs = masterData.coeffs;
            List<string> masterVarNames = masterData.varNames;

            Console.WriteLine("\n--- MASTER Problem Initial Data ---");
            PrintMatrix(masterLHS,"Master LHS");
            PrintArray(masterRHS,"Master RHS");
            PrintArray(masterCoeffs,"Master ObjCoeffs");
            Console.WriteLine("Master Vars: "+string.Join(", ", masterVarNames));

            // Identify subproblem constraints
            List<int> subConsIndices = new List<int>();
            for(int i=0;i<5;i++)
            {
                if(!chosenConsIdx.Contains(i))
                    subConsIndices.Add(i);
            }

            // Identify subproblem variables
            HashSet<string> masterVarSet = new HashSet<string>(masterVarNames);
            List<int> subVarIndices = new List<int>();
            for(int idx=0;idx<allVars.Count; idx++)
            {
                string v= allVars[idx];
                if(!masterVarSet.Contains(v) && v!="theta")
                {
                    subVarIndices.Add(idx);
                }
            }

            // Also check recourse
            bool isCompleteRecourse=true;
            foreach(int ci in chosenConsIdx)
            {
                for(int j=0;j<10;j++)
                {
                    if(Math.Abs(originalLHS[ci,j])>1e-15)
                    {
                        string varName= allVars[j];
                        if(varName.StartsWith("y_")||varName.StartsWith("w_"))
                            isCompleteRecourse=false;
                    }
                }
            }
            Console.WriteLine($"Complete Recourse? => {isCompleteRecourse}");

            // Build scenario multipliers
            double[] scenarioMultipliers = ComputeScenarioMultipliers(numScenarios);

            // Build Master tableau
            var initMaster = BuildMasterTableau(masterLHS, masterRHS, masterCoeffs);
            double[,] masterTableau = initMaster.lhs;
            double[] masterRHSVec = initMaster.rhs;
            double[] masterObjVec = initMaster.obj;

            // find theta index
            int idxTheta = masterVarNames.IndexOf("theta");
            if(idxTheta<0)
            {
                Console.WriteLine("No 'theta' in Master? Benders requires it. Stop.");
                return;
            }

            // L-Shaped iteration
            double UB = double.PositiveInfinity;
            double LB = double.NegativeInfinity;
            double tol=1e-6;
            int maxIter=50;
            bool converged=false;

            Console.WriteLine("\n=== Start L-Shaped Iteration ===");

            for(int iter=1; iter<=maxIter; iter++)
            {
                Console.WriteLine($"\n--- Iteration {iter} ---");
                Console.WriteLine("[Master Tableau LHS]:");
                PrintMatrix(masterTableau,"Master LHS");
                PrintArray(masterRHSVec,"Master RHS");
                PrintArray(masterObjVec,"Master OBJ");

                // Solve Master
                var masterRes = PrimalSimplexSolveFull(masterTableau, masterRHSVec, masterObjVec, debugName:$"MASTER_iter{iter}");
                if(!masterRes.feasible)
                {
                    Console.WriteLine("Master infeasible, stop.");
                    break;
                }

                double[] mSol = masterRes.solution;
                double masterObjVal= masterRes.objVal;

                Console.WriteLine($"Master solution iteration {iter}:");
                for(int i=0;i<masterVarNames.Count;i++)
                    Console.WriteLine($"{masterVarNames[i]} = {mSol[i]:F4}");
                Console.WriteLine($"Master ObjVal= {masterObjVal:F4}");

                UB = Math.Min(UB, masterObjVal);
                double thetaVal = mSol[idxTheta];

                // Solve subproblems
                double totalExpectedCost=0.0;
                bool feasibilityCutNeeded=false;
                var scenarioCuts = new List<(double constantTerm, double[] xCoefs)>();

                for(int s=0;s<numScenarios;s++)
                {
                    double wYield= scenarioMultipliers[s]*2.5;   // wheat yield
                    double cYield= scenarioMultipliers[s]*3.0;   // corn yield
                    double sugarYield= scenarioMultipliers[s]*20.0; // sugar

                    Console.WriteLine($"\n--- Subproblem Scenario {s}, yields=({wYield:F2},{cYield:F2},{sugarYield:F2}) ---");
                    var spRes = SolveSubproblemScenario(mSol, masterVarNames, subConsIndices, subVarIndices,
                                                        wYield,cYield,sugarYield,isCompleteRecourse,
                                                        debugName:$"SUB_s{s}");
                    if(!spRes.feasible)
                    {
                        // infeasible => feasibility cut
                        if(!isCompleteRecourse)
                        {
                            Console.WriteLine($"Scenario {s}: infeasible => add feasibility cut");
                            AddFeasibilityCutFromDual(ref masterTableau, ref masterRHSVec, ref masterObjVec,
                                                      masterVarNames, spRes.dual, spRes.dualRowIndices, subConsIndices);
                            feasibilityCutNeeded=true;
                            break;
                        }
                        else
                        {
                            Console.WriteLine($"Scenario {s}: infeasible but 'complete recourse'? Strange. Stop or cut anyway.");
                            feasibilityCutNeeded=true;
                            break;
                        }
                    }
                    else
                    {
                        // feasible => build optimality cut
                        totalExpectedCost+= spRes.objVal/ numScenarios;
                        var scCut= BuildOptimalityCut(spRes.dual, spRes.dualRowIndices,
                                                      masterVarNames,mSol,
                                                      subConsIndices, wYield,cYield,sugarYield,
                                                      scenarioWeight:1.0);
                        scenarioCuts.Add(scCut);
                    }
                }

                if(feasibilityCutNeeded) continue; // skip adding opt cuts, re-solve next iteration

                // Update LB
                LB= Math.Max(LB, thetaVal);

                if(Math.Abs(UB - LB)<tol)
                {
                    Console.WriteLine($"Converged: UB={UB:F4}, LB={LB:F4}");
                    converged=true;
                    break;
                }

                // Add optimality cuts
                foreach(var scCut in scenarioCuts)
                {
                    double cutConstant= scCut.constantTerm;
                    double[] cutCoefs= scCut.xCoefs;
                    // Evaluate the cut
                    double lhsTheta= mSol[idxTheta];
                    double rhsVal= cutConstant;
                    for(int j=0;j<cutCoefs.Length;j++)
                        rhsVal+= cutCoefs[j]*mSol[j];
                    if(lhsTheta< rhsVal- tol)
                    {
                        Console.WriteLine($"Adding optimality cut: current theta={lhsTheta:F4} < scenario cost expr={rhsVal:F4}");
                        AddOptimalityCutFromDual(ref masterTableau, ref masterRHSVec, ref masterObjVec,
                                                 masterVarNames, cutConstant, cutCoefs, idxTheta);
                    }
                }
            }

            if(!converged)
            {
                Console.WriteLine("Max iteration or not converged, final solve...");
            }

            // Final Solve
            var finalSol = PrimalSimplexSolveFull(masterTableau, masterRHSVec, masterObjVec, debugName:"MASTER_FINAL");
            if(finalSol.feasible)
            {
                Console.WriteLine("\n--- Final Master Solution ---");
                double[] sol = finalSol.solution;
                double fObj = finalSol.objVal;
                for(int i=0;i<masterVarNames.Count;i++)
                {
                    if(!masterVarNames[i].StartsWith("slack_"))
                        Console.WriteLine($"{masterVarNames[i]} = {sol[i]:F4}");
                }
                Console.WriteLine($"Final Objective= {fObj:F4}");
            }
            else
            {
                Console.WriteLine("Final master infeasible or solver error.");
            }

            Console.WriteLine("\n--- End of L-shaped Decomposition ---");
            Console.WriteLine("Check solution above, including subproblem debug prints, for correctness.");
        }

        //======================================================================
        // Build MASTER Problem Data
        //======================================================================
        static (double[,] lhs, double[] rhs, double[] coeffs, List<string> varNames)
        BuildMasterProblemData(HashSet<int> chosenConsIdx)
        {
            int thetaIndex = allVars.IndexOf("theta");
            if(thetaIndex<0)
            {
                Console.WriteLine("No 'theta' found in allVars. Stop.");
                return default;
            }

            HashSet<int> masterVarIndices = new HashSet<int>();
            foreach(int ci in chosenConsIdx)
            {
                for(int j=0;j<10;j++)
                {
                    if(Math.Abs(originalLHS[ci,j])>1e-15)
                        masterVarIndices.Add(j);
                }
            }
            // always include theta
            masterVarIndices.Add(thetaIndex);

            // Also for any master var that is x_1,x_2,x_3 => enforce x>=0 constraints in MASTER
            // We'll do that by adding 3 more constraints: x_i >=0 => -x_i <=0 (only if x_i is a master var)
            // But let's store them after building the chosen constraints.

            List<int> sortedVars = masterVarIndices.ToList();
            sortedVars.Sort();

            int masterCons = chosenConsIdx.Count;
            // We'll also add nonneg constraints for x_i if they appear in MASTER. Let's see how many x_i are in the masterVarIndices
            int xNonnegCount=0;
            List<int> xMasterIndices = new List<int>(); 
            for(int i=0;i<sortedVars.Count;i++)
            {
                int vid= sortedVars[i];
                if(vid==0 || vid==1 || vid==2) // x_1,x_2,x_3 index
                    xMasterIndices.Add(vid);
            }
            xNonnegCount = xMasterIndices.Count;

            int totalConstraints = masterCons + xNonnegCount;
            int totalVars = sortedVars.Count + totalConstraints;  // each row has a slack var

            double[,] masterLHS = new double[totalConstraints, totalVars];
            double[] masterRHS = new double[totalConstraints];
            double[] masterCoeffs = new double[totalVars];
            List<string> masterVarNames = new List<string>();

            // fill the variable name portion
            foreach(int vid in sortedVars)
                masterVarNames.Add(allVars[vid]);

            // add slack var names
            for(int i=0;i<totalConstraints;i++)
            {
                masterVarNames.Add($"slack_{i}");
            }

            // build main constraints
            int row=0;
            foreach(int ci in chosenConsIdx)
            {
                for(int j=0;j<sortedVars.Count;j++)
                {
                    masterLHS[row,j] = originalLHS[ci, sortedVars[j]];
                }
                // slack
                masterLHS[row, sortedVars.Count + row] = 1.0;
                masterRHS[row] = originalRHS[ci];
                row++;
            }

            // build x>=0 constraints if x is in MASTER
            foreach(int xvid in xMasterIndices)
            {
                // row: -xvid <=0
                for(int j=0;j<sortedVars.Count;j++)
                {
                    masterLHS[row,j] = 0.0;
                }
                int idxInSorted = sortedVars.IndexOf(xvid);
                masterLHS[row, idxInSorted] = -1.0; // -x_i <=0
                masterLHS[row, sortedVars.Count + row] = 1.0; // slack
                masterRHS[row] = 0.0;
                row++;
            }

            // build objective
            for(int j=0;j<sortedVars.Count;j++)
            {
                string vname = allVars[sortedVars[j]];
                masterCoeffs[j] = varObj[vname];
            }
            // slack cost=0

            return (masterLHS,masterRHS,masterCoeffs,masterVarNames);
        }

        static (double[,] lhs, double[] rhs, double[] obj) BuildMasterTableau(double[,] masterLHS, double[] masterRHS, double[] masterCoeffs)
        {
            int m = masterRHS.Length;
            int n = masterCoeffs.Length;
            double[,] lhs = new double[m,n];
            double[] rhs = new double[m];
            double[] obj = new double[n];

            for(int i=0;i<m;i++)
            {
                for(int j=0;j<n;j++)
                {
                    lhs[i,j]= masterLHS[i,j];
                }
                rhs[i]= masterRHS[i];
            }
            for(int j=0;j<n;j++)
                obj[j]= masterCoeffs[j];

            return (lhs,rhs,obj);
        }

        //======================================================================
        // SUBPROBLEM
        //======================================================================
        public struct SubproblemResult
        {
            public bool feasible;
            public double objVal;
            public double[] dual;
            public int[] dualRowIndices;
        }

        static SubproblemResult SolveSubproblemScenario(double[] masterSolVal,
            List<string> masterVarNames,
            List<int> subConsIndices,
            List<int> subVarIndices,
            double wheatYield, double cornYield, double sugarYield,
            bool isCompleteRecourse,
            string debugName="SUB")
        {
            // Build subproblem with constraints subConsIndices + nonneg for y_1,y_2,w_1,w_2,w_3,w_4
            // Each subproblem constraint is ≤ form with slack.
            int baseM = subConsIndices.Count;
            // We'll add 6 nonneg constraints for (y1,y2,w1,w2,w3,w4). Each is -variable <=0
            // But only for subVarIndices that correspond to y_/w_.
            // subVarIndices are the set that is not in master + not theta.

            List<int> nonnegRows = new List<int>();
            List<int> nnegVarIndex = new List<int>();
            foreach(int vid in subVarIndices)
            {
                // y1,y2,w1,w2,w3,w4 => must be >=0
                string vname = allVars[vid];
                if(vname.StartsWith("y_") || vname.StartsWith("w_"))
                {
                    nnegVarIndex.Add(vid);
                }
            }
            int nncount = nnegVarIndex.Count;
            int subM = baseM + nncount;  
            int subVarCount = subVarIndices.Count;
            int totalVars = subVarCount + subM;  // each row has a slack

            double[,] subLHS = new double[subM, totalVars];
            double[] subRHS = new double[subM];
            double[] subObj = new double[totalVars];

            // fill subObj for subVarIndices
            for(int j=0;j<subVarCount;j++)
            {
                string v = allVars[subVarIndices[j]];
                subObj[j] = varObj[v];
            }

            // fill the base constraints
            for(int i=0; i<baseM; i++)
            {
                int ci = subConsIndices[i];
                double[] row = new double[10];
                for(int k=0;k<10;k++)
                    row[k]= originalLHS[ci,k];
                double rhsVal= originalRHS[ci];

                // adjust yields
                if(ci==1) row[0]= -wheatYield;  // C2
                if(ci==2) row[1]= -cornYield;   // C3
                if(ci==3) row[2]= -sugarYield;  // C4

                // sub out master var solution
                for(int mv=0; mv< masterVarNames.Count; mv++)
                {
                    string mvn= masterVarNames[mv];
                    int gid= allVars.IndexOf(mvn);
                    if(gid>=0 && Math.Abs(row[gid])>1e-15)
                    {
                        rhsVal -= row[gid]* masterSolVal[mv];
                        row[gid]=0.0;
                    }
                }

                // fill subLHS row
                for(int j=0;j<subVarCount;j++)
                {
                    int gidx= subVarIndices[j];
                    subLHS[i,j]= row[gidx];
                }
                // slack
                subLHS[i, subVarCount + i]= 1.0;
                subRHS[i]= rhsVal;
            }

            // add nonneg constraints for y_i,w_i => -var <=0
            for(int idx=0; idx<nncount; idx++)
            {
                int vid= nnegVarIndex[idx];
                int rowID= baseM + idx;  // offset
                for(int j=0;j<subVarCount;j++)
                {
                    subLHS[rowID,j]=0.0;
                }
                int colInSubVars= subVarIndices.IndexOf(vid);
                subLHS[rowID,colInSubVars]= -1.0;
                // slack
                subLHS[rowID, subVarCount + rowID] = 1.0;
                subRHS[rowID]= 0.0;
            }

            var subTbl = BuildSubproblemTableau(subLHS, subRHS, subObj);
            var spRes = SolveSubproblemPrimalWithDual(subTbl.lhs, subTbl.rhs, subTbl.obj, debugName);

            SubproblemResult ret = new SubproblemResult();
            ret.feasible = spRes.feasible;
            ret.objVal = spRes.objVal;
            ret.dual = spRes.duals;
            // The first "baseM" dual multipliers correspond to subConsIndices
            ret.dualRowIndices = subConsIndices.ToArray();
            // The non-neg constraints do not directly produce Benders cuts for feasibility/optimality 
            // (they're not scenario-based constraints). So we only store the first "baseM" dual row indices if needed.

            return ret;
        }

        static (double[,] lhs, double[] rhs, double[] obj) BuildSubproblemTableau(double[,] subLHS, double[] subRHS, double[] subObj)
        {
            int m= subRHS.Length;
            int n= subObj.Length;
            double[,] lhs = new double[m,n];
            double[] rhs = new double[m];
            double[] obj = new double[n];
            for(int i=0;i<m;i++)
            {
                for(int j=0;j<n;j++)
                {
                    lhs[i,j]= subLHS[i,j];
                }
                rhs[i]= subRHS[i];
            }
            for(int j=0;j<n;j++)
                obj[j]= subObj[j];
            return (lhs,rhs,obj);
        }

        public struct SimplexResult
        {
            public bool feasible;
            public double[] solution;
            public double objVal;
            public double[] duals;
        }

        // Solve subproblem primal with dual
        static SimplexResult SolveSubproblemPrimalWithDual(double[,] inputLHS, double[] inputRHS, double[] inputObj, string debugName="SUB")
        {
            Console.WriteLine($"\n[{debugName}] Subproblem LHS:");
            PrintMatrix(inputLHS,"Sub-LHS");
            PrintArray(inputRHS,"Sub-RHS");
            PrintArray(inputObj,"Sub-Obj");

            var primalRes= PrimalSimplexSolveFull(inputLHS, inputRHS, inputObj, debugName:debugName);
            return primalRes;
        }

        //======================================================================
        // PRIMAL Simplex 
        //======================================================================
        static SimplexResult PrimalSimplexSolveFull(double[,] inputLHS, double[] inputRHS, double[] inputObj, string debugName="MASTER")
        {
            int m= inputRHS.Length;
            int n= inputObj.Length;

            double[,] A = (double[,])inputLHS.Clone();
            double[] B = (double[])inputRHS.Clone();
            double[] c = (double[])inputObj.Clone();

            int[] basicVar = new int[m];
            for(int i=0;i<m;i++)
            {
                basicVar[i] = n - m + i;  // slack columns in basis
            }

            double EPS=1e-9;
            bool done=false;
            int iteration=0;

            Console.WriteLine($"\n*** {debugName}: Start PRIMAL Simplex ***");
            while(!done)
            {
                iteration++;
                // compute reduced costs
                double[] reducedCosts= new double[n];
                for(int j=0;j<n;j++)
                {
                    reducedCosts[j] = c[j];
                    for(int i=0;i<m;i++)
                    {
                        reducedCosts[j] -= c[basicVar[i]]* A[i,j];
                    }
                }

                // print iteration
                Console.WriteLine($"\n{debugName} Iteration {iteration} BFS:");
                Console.WriteLine("B= "+string.Join(", ", B.Select(x=>x.ToString("F4"))));
                Console.WriteLine("c= "+string.Join(", ", c.Select(x=>x.ToString("F4"))));
                Console.WriteLine("ReducedCosts= "+string.Join(", ", reducedCosts.Select(x=>x.ToString("F4"))));

                // find entering var
                int entering=-1;
                for(int j=0;j<n;j++)
                {
                    if(reducedCosts[j]< -EPS)
                    {
                        entering=j; 
                        break;
                    }
                }
                if(entering<0)
                {
                    // optimal
                    done=true;
                    break;
                }

                // ratio test
                int leaving=-1;
                double minRatio= double.MaxValue;
                for(int i=0;i<m;i++)
                {
                    if(A[i,entering]>EPS)
                    {
                        double ratio= B[i]/ A[i,entering];
                        if(ratio< minRatio)
                        {
                            minRatio= ratio;
                            leaving= i;
                        }
                    }
                }
                if(leaving<0)
                {
                    Console.WriteLine($"{debugName}: Unbounded primal. Return infeasible or large cost.");
                    return new SimplexResult { feasible=false, solution=null, objVal=double.PositiveInfinity, duals=null };
                }

                // pivot
                double pivot= A[leaving,entering];
                Console.WriteLine($"{debugName}: Pivot => entering={entering}, leaving row={leaving}, pivot={pivot:F4}");

                // normalize pivot row
                for(int j=0;j<n;j++)
                    A[leaving,j]/= pivot;
                B[leaving]/= pivot;

                double pivotCost= c[entering];
                for(int j=0;j<n;j++)
                    c[j] -= pivotCost*A[leaving,j];

                // update other rows
                for(int i=0;i<m;i++)
                {
                    if(i!=leaving)
                    {
                        double factor= A[i,entering];
                        for(int j=0;j<n;j++)
                        {
                            A[i,j] -= factor*A[leaving,j];
                        }
                        B[i]-= factor*B[leaving];
                    }
                }

                basicVar[leaving]= entering;
            }

            // build final solution
            double[] solution= new double[n];
            for(int i=0;i<m;i++)
            {
                solution[basicVar[i]]= B[i];
            }

            double objVal=0.0;
            for(int j=0;j<n;j++)
                objVal += solution[j]* inputObj[j];

            // dual multipliers
            double[] duals= new double[m];
            for(int i=0;i<m;i++)
            {
                duals[i]= c[basicVar[i]];
            }

            Console.WriteLine($"\n{debugName}: PRIMAL SIMPLEX DONE. ObjVal={objVal:F4}");
            Console.WriteLine($"{debugName}: Final solution= {string.Join(", ", solution.Select(x=>x.ToString("F4")))}");
            Console.WriteLine($"{debugName}: Dual multipliers= {string.Join(", ", duals.Select(x=>x.ToString("F4")))}");

            return new SimplexResult { feasible=true, solution=solution, objVal=objVal, duals=duals };
        }

        //======================================================================
        // Benders Cut Builders
        //======================================================================
        static (double constantTerm, double[] xCoefs) BuildOptimalityCut(
            double[] scenarioDuals,
            int[] subConsIdx,
            List<string> masterVarNames,
            double[] masterSol,
            List<int> subConsIndices,
            double wYield, double cYield, double sugarYield,
            double scenarioWeight=1.0)
        {
            // subproblem constraints are in ≤ form. scenarioDuals[i] is the dual for subConsIdx[i].
            double constTerm=0.0;
            double[] xCoefs = new double[masterVarNames.Count];

            for(int row=0; row< scenarioDuals.Length; row++)
            {
                double dualVal = scenarioDuals[row]*scenarioWeight;
                if(Math.Abs(dualVal)<1e-12) continue;
                int cIndex= subConsIdx[row];
                // skip if row≥ subConsIndices.Count => might be a nonneg constraint row
                if(row>= subConsIndices.Count) continue;  // the subproblem's nonneg constraints do not generate benders cuts.

                double rhs= originalRHS[cIndex];
                double[] cRow = new double[10];
                for(int k=0;k<10;k++)
                    cRow[k]= originalLHS[cIndex,k];

                // yield adjust
                if(cIndex==1) cRow[0]= -wYield;
                if(cIndex==2) cRow[1]= -cYield;
                if(cIndex==3) cRow[2]= -sugarYield;

                double rowConst= rhs;
                // separate out master var portion
                for(int mv=0; mv<masterVarNames.Count; mv++)
                {
                    string mvn= masterVarNames[mv];
                    int gid= allVars.IndexOf(mvn);
                    if(gid>=0 && Math.Abs(cRow[gid])>1e-15)
                    {
                        xCoefs[mv]+= dualVal*cRow[gid];
                        rowConst-= cRow[gid]* masterSol[mv];
                    }
                }
                constTerm+= dualVal* rowConst;
            }
            return (constTerm, xCoefs);
        }

        static void AddOptimalityCutFromDual(ref double[,] masterLHS, ref double[] masterRHS, ref double[] masterObj,
                                             List<string> masterVarNames, 
                                             double constantTerm, double[] xCoefs,
                                             int idxTheta)
        {
            int oldCons= masterRHS.Length;
            int oldVars= masterObj.Length;

            double[,] newLHS= new double[oldCons+1, oldVars];
            double[] newRHS= new double[oldCons+1];
            for(int i=0;i<oldCons;i++)
            {
                for(int j=0;j<oldVars;j++)
                    newLHS[i,j]= masterLHS[i,j];
                newRHS[i]= masterRHS[i];
            }

            // new row:  theta >= constTerm + sum(xCoefs*x)
            // => -theta + sum(xCoefs_j*x_j) <= -constTerm
            for(int j=0;j<oldVars;j++)
                newLHS[oldCons,j]= 0.0;
            newLHS[oldCons, idxTheta]= -1.0;
            for(int mv=0;mv<xCoefs.Length;mv++)
            {
                newLHS[oldCons,mv] = xCoefs[mv];
            }
            newRHS[oldCons]= -constantTerm;

            masterLHS= newLHS;
            masterRHS= newRHS;
        }

        static void AddFeasibilityCutFromDual(ref double[,] masterLHS, ref double[] masterRHS, ref double[] masterObj,
                                              List<string> masterVarNames,
                                              double[] scenarioDuals, int[] subConsIdx, List<int> allSubConsIdx)
        {
            Console.WriteLine("Adding Feasibility Cut from Farkas ray...");
            int oldCons= masterRHS.Length;
            int oldVars= masterObj.Length;

            double[,] newLHS= new double[oldCons+1, oldVars];
            double[] newRHS= new double[oldCons+1];
            for(int i=0;i<oldCons;i++)
            {
                for(int j=0;j<oldVars;j++)
                    newLHS[i,j]= masterLHS[i,j];
                newRHS[i]= masterRHS[i];
            }

            double[] cutRow= new double[oldVars];
            double rhsCut=0.0;

            for(int i=0; i<scenarioDuals.Length; i++)
            {
                double dualVal= scenarioDuals[i];
                if(Math.Abs(dualVal)<1e-15) continue;
                if(i>= allSubConsIdx.Count) continue; // skip nonneg constraints

                int cIndex = subConsIdx[i];
                double rhsVal= originalRHS[cIndex];
                double[] row= new double[10];
                for(int k=0;k<10;k++)
                    row[k]= originalLHS[cIndex,k];

                rhsCut += dualVal* rhsVal;
                // master portion
                for(int mv=0; mv<masterVarNames.Count; mv++)
                {
                    string mvn= masterVarNames[mv];
                    int gid= allVars.IndexOf(mvn);
                    if(gid>=0)
                    {
                        cutRow[mv] += dualVal*row[gid];
                    }
                }
            }

            // want cutRow*x >= rhsCut => -cutRow*x <= -rhsCut
            for(int j=0;j<oldVars;j++)
                newLHS[oldCons,j]= -cutRow[j];
            newRHS[oldCons] = -rhsCut;

            masterLHS= newLHS;
            masterRHS= newRHS;
        }

        //======================================================================
        // Utilities
        //======================================================================
        static double[] ComputeScenarioMultipliers(int numScenarios)
        {
            double[] arr = new double[numScenarios];
            if(numScenarios==1)
            {
                arr[0]=1.0; 
                return arr;
            }
            double step=0.4/(numScenarios-1); // from 0.8 to 1.2
            for(int i=0; i<numScenarios; i++)
            {
                arr[i] = 0.8 + i*step;
            }
            return arr;
        }

        static void PrintOriginalProblem()
        {
            Console.WriteLine("\n--- Farmer Problem Statement ---");
            Console.WriteLine("Minimize: 150x_1 +230x_2 +260x_3 +238y_1 +210y_2 -170w_1 -150w_2 -36w_3 -10w_4");
            Console.WriteLine("Subject to:");
            Console.WriteLine("C1: x_1 + x_2 + x_3 <= 500");
            Console.WriteLine("C2: 2.5x_1 + y_1 - w_1 >= 200 => -2.5x_1 - y_1 + w_1 <= -200");
            Console.WriteLine("C3: 3x_2 + y_2 - w_2 >= 240 => -3x_2 - y_2 + w_2 <= -240");
            Console.WriteLine("C4: w_3 + w_4 <= 20x_3 => w_3 + w_4 -20x_3 <= 0");
            Console.WriteLine("C5: w_3 <= 6000");
            Console.WriteLine("x_i >= 0, y_i >= 0, w_i >= 0. Total land=500 acres.");
            Console.WriteLine("Second-stage yields vary from 80% to 120% of nominal yields [2.5,3,20].");
        }

        static void PrintMatrix(double[,] mat, string label="Matrix")
        {
            Console.WriteLine($"{label} [{mat.GetLength(0)}x{mat.GetLength(1)}]:");
            for(int i=0;i<mat.GetLength(0);i++)
            {
                for(int j=0;j<mat.GetLength(1);j++)
                {
                    Console.Write($"{mat[i,j]:F2}\t");
                }
                Console.WriteLine();
            }
        }

        static void PrintArray(double[] arr, string label="Array")
        {
            Console.WriteLine($"{label}: {string.Join(", ", arr.Select(x=>x.ToString("F4")))}");
        }
    }
}
