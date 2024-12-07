using System;
using System.Collections.Generic;
using System.Linq;

// Problem data
double totalLand = 500;
Dictionary<string, double> yields = new Dictionary<string, double> { { "wheat", 2.5 }, { "corn", 3.0 }, { "sugar", 20.0 } };
Dictionary<string, double> plantingCosts = new Dictionary<string, double> { { "wheat", 150 }, { "corn", 230 }, { "sugar", 260 } };
Dictionary<string, double> sellingPrices = new Dictionary<string, double> { { "wheat", 170 }, { "corn", 150 }, { "sugar_favorable", 36 }, { "sugar_unfavorable", 10 } };
Dictionary<string, double> purchasePrices = new Dictionary<string, double> { { "wheat", 238 }, { "corn", 210 } };
Dictionary<string, double> minReq = new Dictionary<string, double> { { "wheat", 200 }, { "corn", 240 } };

class Program
{
    static void Main(string[] args)
    {
        // User input: master variables
        Console.WriteLine("Enter variables for master problem (e.g. x_wheat,x_corn,x_sugar,y_11,w_11):");
        string[] masterVarsInput = Console.ReadLine().Split(',');
        HashSet<string> masterVars = new HashSet<string>(masterVarsInput.Select(v => v.Trim()).Where(v => v != ""));

        // Always add theta
        masterVars.Add("theta");

        // User input: number of scenarios
        Console.WriteLine("Enter number of scenarios:");
        int scenarioCount = Convert.ToInt32(Console.ReadLine());

        // Generate scenario yield multipliers uniformly in [0.8,1.2]
        double[] scenarioMultipliers = new double[scenarioCount];
        if (scenarioCount == 1)
        {
            scenarioMultipliers[0] = 1.0;
        }
        else
        {
            for (int i = 0; i < scenarioCount; i++)
            {
                scenarioMultipliers[i] = 0.8 + 0.4 * i / (scenarioCount - 1);
            }
        }

        // Determine if complete recourse
        bool completeRecourse = true;
        foreach (var v in masterVars)
        {
            if (v.StartsWith("y_") || v.StartsWith("w_"))
            {
                completeRecourse = false;
                break;
            }
        }

        // Master constraints
        HashSet<string> masterConstraints = new HashSet<string>();
        Console.WriteLine("Enter constraints for master (e.g. LandAllocation), leave blank if none:");
        string[] masterConsInput = Console.ReadLine().Split(',');
        foreach (var c in masterConsInput)
        {
            string cc = c.Trim();
            if (cc != "") masterConstraints.Add(cc);
        }
        if (masterConstraints.Count == 0)
        {
            masterConstraints.Add("LandAllocation");
        }

        MasterProblem master = BuildInitialMaster(masterVars, masterConstraints, totalLand, plantingCosts);

        int maxIterations = 50;
        double LB = Double.NegativeInfinity;
        double UB = Double.PositiveInfinity;
        double tolerance = 1e-6;

        // We solve one subproblem per iteration. Let's pick scenarios round-robin:
        int scenarioIndex = 0;

        for (int iteration = 1; iteration <= maxIterations; iteration++)
        {
            (bool masterFeasible, double[] masterSol, double masterObj) = SolveMasterProblem(master);
            if (!masterFeasible)
            {
                Console.WriteLine("Master infeasible. Stopping.");
                break;
            }

            LB = masterObj;
            bool anyCutAdded = false;

            // Solve one subproblem for scenarioIndex
            int s = scenarioIndex;
            scenarioIndex = (scenarioIndex + 1) % scenarioCount;

            // Solve subproblem
            (bool feasibleSP, double scenarioCost, double[] duals, bool farkas) =
                SolveSubproblem(masterSol, yields, sellingPrices, purchasePrices, minReq, scenarioMultipliers[s], scenarioMultipliers[s], scenarioMultipliers[s], master);

            if (!feasibleSP && !completeRecourse)
            {
                // Feasibility cut
                BendersCut feasCut = GenerateFeasibilityCut(duals, farkas, master);
                master.AddCut(feasCut);
                anyCutAdded = true;
            }
            else if (feasibleSP)
            {
                // Optimality cut
                BendersCut optCut = GenerateOptimalityCut(duals, scenarioCost, master);
                master.AddCut(optCut);
                anyCutAdded = true;

                // If feasible, we have a candidate UB
                double thetaVal = ExtractThetaValue(master, master.LastSolution);
                // scenarioCost here is single scenario cost. We have only one scenario considered per iteration.
                // For a more accurate UB estimate, you would need the expected cost over all scenarios.
                // Without recomputing all scenarios, let's assume scenarioCost approximates second-stage:
                UB = Math.Min(UB, LB + (scenarioCost - thetaVal));
            }

            if (!anyCutAdded && Math.Abs(UB - LB) < tolerance)
            {
                Console.WriteLine("Converged: UB ~ LB");
                break;
            }
        }

        Console.WriteLine("Final Master Solution:");
        PrintSolution(master, master.LastSolution);
    }

    static MasterProblem BuildInitialMaster(HashSet<string> masterVars, HashSet<string> masterCons,
        double totalLand, Dictionary<string,double> plantingCosts)
    {
        MasterProblem mp = new MasterProblem();
        mp.Variables = masterVars.ToList();
        int numVars = mp.Variables.Count;

        mp.ObjCoeffs = new double[numVars];
        for (int i = 0; i < numVars; i++)
        {
            string v = mp.Variables[i];
            if (v=="theta") mp.ObjCoeffs[i]=1.0;
            else if (v=="x_wheat") mp.ObjCoeffs[i]=-plantingCosts["wheat"];
            else if (v=="x_corn") mp.ObjCoeffs[i]=-plantingCosts["corn"];
            else if (v=="x_sugar") mp.ObjCoeffs[i]=-plantingCosts["sugar"];
            else mp.ObjCoeffs[i]=0.0;
        }

        List<double[]> lhs = new List<double[]>();
        List<double> rhs = new List<double>();
        List<char> sense = new List<char>();

        if (masterCons.Contains("LandAllocation"))
        {
            double[] row = new double[numVars];
            for (int j=0;j<numVars;j++)
            {
                string varName = mp.Variables[j];
                if (varName.StartsWith("x_")) row[j]=1.0;
                else row[j]=0.0;
            }
            lhs.Add(row);
            rhs.Add(totalLand);
            sense.Add('L');
        }

        mp.LHS = lhs.ToArray();
        mp.RHS = rhs.ToArray();
        mp.Sense = sense.ToArray();

        return mp;
    }

    static (bool feasible, double[] solution, double objVal) SolveMasterProblem(MasterProblem mp)
    {
        return SolveUsingPrimalSimplex(mp.ObjCoeffs, mp.LHS, mp.RHS, mp.Sense, mp.Variables.Count, mp.LHS.Length);
    }

    static (bool feasible, double scenarioCost, double[] duals, bool farkas) SolveSubproblem(
        double[] masterSol,
        Dictionary<string,double> yields,
        Dictionary<string,double> sellingPrices,
        Dictionary<string,double> purchasePrices,
        Dictionary<string,double> minReq,
        double yw, double yc, double ys,
        MasterProblem mp)
    {
        // Extract x's:
        double xw=0, xc=0, xs=0;
        for (int i=0;i<mp.Variables.Count;i++)
        {
            string v=mp.Variables[i];
            if (v=="x_wheat") xw=masterSol[i];
            else if (v=="x_corn") xc=masterSol[i];
            else if (v=="x_sugar") xs=masterSol[i];
        }

        double wheatProd = xw*yields["wheat"]*yw;
        double cornProd = xc*yields["corn"]*yc;
        double sugarProd = xs*yields["sugar"]*ys;

        // Build subproblem LP arrays:
        // This is a placeholder. In reality, form second-stage constraints from problem statement.
        // Solve with your LP solver. Extract duals.
        bool feasible = true;
        double scenarioProfit = 1000.0; // dummy
        double[] duals = new double[]{0.5};
        bool farkas = false;
        return (feasible, scenarioProfit, duals, farkas);
    }

    static BendersCut GenerateFeasibilityCut(double[] duals, bool farkas, MasterProblem mp)
    {
        BendersCut c = new BendersCut();
        // Dummy feasibility cut
        for (int i=0;i<mp.Variables.Count;i++)
        {
            string v=mp.Variables[i];
            if (v.StartsWith("x_")) c.Coeffs[v]=-1.0;
            else if (v=="theta") c.Coeffs[v]=0.0;
        }
        c.RHS = -100.0;
        c.Sense='G';
        return c;
    }

    static BendersCut GenerateOptimalityCut(double[] duals, double scenarioCost, MasterProblem mp)
    {
        BendersCut c = new BendersCut();
        c.Coeffs["theta"]=1.0;
        if (mp.Variables.Contains("x_wheat")) c.Coeffs["x_wheat"]=-0.5;
        if (mp.Variables.Contains("x_corn")) c.Coeffs["x_corn"]=-0.3;
        if (mp.Variables.Contains("x_sugar")) c.Coeffs["x_sugar"]=0.0;
        c.RHS = scenarioCost -50.0;
        c.Sense='G';
        return c;
    }

    static double ExtractThetaValue(MasterProblem mp, double[] sol)
    {
        for (int i=0;i<mp.Variables.Count;i++)
        {
            if (mp.Variables[i]=="theta") return sol[i];
        }
        return 0.0;
    }

    static void PrintSolution(MasterProblem mp, double[] sol)
    {
        if (sol==null) return;
        for (int i=0;i<mp.Variables.Count;i++)
        {
            Console.WriteLine($"{mp.Variables[i]}={sol[i]:F4}");
        }
    }

    // Replace with your primal simplex from midterm
    static (bool feasible, double[] solution, double objVal) SolveUsingPrimalSimplex(double[] objCoeffs, double[][] lhs, double[] rhs, char[] sense, int numVars, int numCons)
    {
        double[] sol = new double[numVars];
        for (int i=0;i<numVars;i++) sol[i]=100.0; // dummy
        double objVal=500.0; // dummy
        return (true,sol,objVal);
    }
}

class MasterProblem
{
    public List<string> Variables;
    public double[] ObjCoeffs;
    public double[][] LHS;
    public double[] RHS;
    public char[] Sense;
    public List<BendersCut> Cuts = new List<BendersCut>();
    public double[] LastSolution;

    public void AddCut(BendersCut c)
    {
        int oldRows = RHS.Length;
        int oldCols = ObjCoeffs.Length;
        double[][] newLHS = new double[oldRows+1][];
        for (int i=0;i<oldRows;i++)
        {
            newLHS[i]=new double[oldCols];
            for (int j=0;j<oldCols;j++)
                newLHS[i][j]=LHS[i][j];
        }
        newLHS[oldRows]=new double[oldCols];
        for (int j=0;j<oldCols;j++)
        {
            string varName=Variables[j];
            double val=0.0;
            if (c.Coeffs.ContainsKey(varName))
                val=c.Coeffs[varName];
            newLHS[oldRows][j]=val;
        }

        double[] newRHS=new double[oldRows+1];
        for (int i=0;i<oldRows;i++) newRHS[i]=RHS[i];
        newRHS[oldRows]=c.RHS;

        char[] newSense=new char[oldRows+1];
        for (int i=0;i<oldRows;i++) newSense[i]=Sense[i];
        newSense[oldRows]=c.Sense;

        LHS=newLHS;
        RHS=newRHS;
        Sense=newSense;
        Cuts.Add(c);
    }
}

class BendersCut
{
    public Dictionary<string,double> Coeffs=new Dictionary<string,double>();
    public double RHS;
    public char Sense;
}
