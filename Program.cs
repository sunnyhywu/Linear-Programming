using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace StochasticFarmerProblem
{
    class BendersDecomposition
    {

        static void Main(string[] args)
        {
            // 1. Prompt the user to choose the number of scenarios
            Console.WriteLine("Enter the number of scenarios:");

            int numScenarios;
            while (!int.TryParse(Console.ReadLine(), out numScenarios) || numScenarios <= 0)
            {
                Console.WriteLine("Invalid input. Please enter a positive integer for the number of scenarios.");
            }

            // 2. Generate the list of variables based on the number of scenarios
            List<string> allVariablesList = GenerateVariables(numScenarios);
            string[] allVariables = allVariablesList.ToArray();

            Console.WriteLine($"\nAvailable variables for {numScenarios} scenarios:");
            Console.WriteLine(string.Join(" ", allVariables));

            Console.WriteLine(); // Add a line break for better readability

            // 3. Construct the objective coefficient matrix
            double[] objectiveCoefficients = BuildObjectiveCoefficients(allVariables, numScenarios);
            // Generate LHS matrix with coefficients varying by uniform distribution
            double[,] lhsMatrix = GenerateLhsMatrix(allVariables, numScenarios);
            double[] rhsMatrix = GenerateRhsMatrix(numScenarios);

            // 4. Prompt the user to choose variables for the master problem
            Console.WriteLine("\nEnter the variables to include in the master problem (e.g., x_1,x_2,x_3,w_1_1):");
            string inputVariables = Console.ReadLine();
            // Parse user input into a list of variable names
            string[] masterVariables = inputVariables.Split(',');
            Console.WriteLine("\nSelected variables for master problem:");
            foreach (string variable in masterVariables)
            {
                Console.WriteLine(variable.Trim());
            }

            // 5. Extract Master and Subproblem data
            var (masterCoefficients, subproblemCoefficients, masterLhs, masterRhs,
                subproblemLhs, subproblemRhs, masterConstraints, masterTotalVars) =
                    ExtractMasterAndSubproblemData(
                        allVariables,
                        objectiveCoefficients,
                        lhsMatrix,
                        rhsMatrix,
                        masterVariables);


            //6. L-shaped

            // Benders parameters
            double UB = double.PositiveInfinity;
            double LB = double.NegativeInfinity;
            int iteration = 0;
            double tolerance = 1e-6;
            int maxIterations = 50;

            // Determine complete recourse
            bool isCompleteRecourse = true;
            foreach (var mv in masterVariables)
            {
                if (mv.StartsWith("y_") || mv.StartsWith("w_"))
                {
                    isCompleteRecourse = false;
                    break;
                }
            }

            // Generate scenario yield multipliers uniformly in [0.8,1.2]
            double[] scenarioMultipliers = new double[numScenarios];
            if (numScenarios == 1)
            {
                scenarioMultipliers[0] = 1.0;
            }
            else
            {
                for (int i = 0; i < numScenarios; i++)
                {
                    scenarioMultipliers[i] = 0.8 + 0.4 * i / (numScenarios - 1);
                }
            }

            // Solve one subproblem per iteration, round-robin
            int scenarioIndex = 0;

            while (Math.Abs(UB - LB) > tolerance && iteration < maxIterations)
            {
                iteration++;
                Console.WriteLine($"\n--- Benders Iteration {iteration} ---");

                // Solve master problem
                // We have masterCoefficients, masterLhs, masterRhs
                // Use primal simplex
                (bool feasibleMaster, double[] masterSol, double masterObj) =
                    SolveUsingPrimalSimplex(masterCoefficients, masterLhs, masterRhs, numMasterVars, numMasterCons);

                if (!feasibleMaster)
                {
                    Console.WriteLine("Master infeasible. Stopping.");
                    break;
                }

                LB = masterObj; // Master provides LB

                // Solve one scenario subproblem
                int s = scenarioIndex;
                scenarioIndex = (scenarioIndex + 1) % numScenarios;

                // Solve subproblem to get scenario cost and duals for cuts
                (bool feasibleSP, double scenarioCost, double[] duals, bool farkas) =
                    SolveSubproblemScenario(masterSol, masterVarNames, scenarioMultipliers[s]);

                bool anyCutAdded = false;

                if (!feasibleSP && !isCompleteRecourse)
                {
                    // Add feasibility cut from duals
                    AddFeasibilityCut(ref masterLhs, ref masterRhs, ref masterCoefficients, masterVarNames, duals);
                    anyCutAdded = true;
                    Console.WriteLine("Feasibility cut added to master.");
                }
                else if (feasibleSP)
                {
                    // Add optimality cut from duals
                    AddOptimalityCut(ref masterLhs, ref masterRhs, ref masterCoefficients, masterVarNames, scenarioCost, duals);
                    anyCutAdded = true;
                    Console.WriteLine("Optimality cut added to master.");

                    // Update UB
                    double thetaVal = ExtractThetaValue(masterSol, masterVarNames);
                    // Typically we need expected second-stage cost over all scenarios, but here we approximate with scenarioCost
                    // For a full solution, you would re-evaluate all scenarios to get a true UB.
                    UB = Math.Min(UB, LB + (scenarioCost - thetaVal));
                }

                if (!anyCutAdded && Math.Abs(UB - LB) < tolerance)
                {
                    Console.WriteLine("Converged: UB ~ LB");
                    break;
                }
            }

            Console.WriteLine("\nFinal Master Solution:");
            PrintSolution(masterVarNames, SolveUsingPrimalSimplex(masterCoefficients, masterLhs, masterRhs, numMasterVars, numMasterCons).solution);
        }

        // Generate variable names for the farmer's problem
        // Include first-stage variables: x_wheat, x_corn, x_sugar and theta
        // Second-stage variables (y,w) can be added if needed.
        static List<string> GenerateVariables(int numScenarios)
        {
            List<string> vars = new List<string> { "theta", "x_wheat", "x_corn", "x_sugar" };
            // If your problem requires scenario-dependent variables (y_ij, w_ij), add them here
            // For example:
            for (int s=1; s<=numScenarios; s++) {
                vars.Add($"y_wheat_{s}");
                vars.Add($"y_corn_{s}");
                vars.Add($"w_sugar_high_{s}");
                vars.Add($"w_sugar_low_{s}");
            }
            return vars;
        }

        // Build objective coefficients based on the farmer's problem data
        // The objective: Maximize (theta) - sum(plantingCosts*x)
        static double[] BuildObjectiveCoefficients(string[] allVars, int numScenarios)
        {
            double[] obj = new double[allVars.Length];
            for (int i = 0; i < allVars.Length; i++)
            {
                string v = allVars[i];
                if (v == "theta") obj[i] = 1.0;
                else if (v == "x_wheat") obj[i] = -150;
                else if (v == "x_corn") obj[i] = -230;
                else if (v == "x_sugar") obj[i] = -260;
                else obj[i] = 0.0; // Adjust for second-stage vars if any appear
            }
            return obj;
        }

        // Generate LHS matrix for the constraints
        // For example: x_wheat + x_corn + x_sugar <= 500 (land constraint)
        static double[,] GenerateLhsMatrix(string[] allVars, int numScenarios)
        {
            int numCons = 1; // Land constraint
            int numVars = allVars.Length;
            double[,] lhs = new double[numCons, numVars];
            for (int j = 0; j < numVars; j++)
            {
                if (allVars[j].StartsWith("x_"))
                    lhs[0, j] = 1.0;
                else
                    lhs[0, j] = 0.0;
            }
            return lhs;
        }

        // Generate RHS for the constraints
        // land: <=500
        static double[] GenerateRhsMatrix(int numScenarios)
        {
            return new double[] { 500 };
        }

        // Extract master and subproblem data
        // In a real problem, separate master variables/constraints from second-stage ones.
        static (double[] masterCoefficients,
                double[] subproblemCoefficients,
                double[,] masterLhs,
                double[] masterRhs,
                double[,] subproblemLhs,
                double[] subproblemRhs,
                int numMasterCons,
                int numMasterVars,
                string[] masterVarNames)
            ExtractMasterAndSubproblemData(
            string[] allVariables,
            double[] objCoeffs,
            double[,] lhs,
            double[] rhs,
            string[] masterVars)
        {
            // Identify indices of master variables
            Dictionary<string, int> varIndex = new Dictionary<string, int>();
            for (int i = 0; i < allVariables.Length; i++) varIndex[allVariables[i]] = i;

            List<int> masterVarIndices = new List<int>();
            foreach (var mv in masterVars)
            {
                if (varIndex.ContainsKey(mv))
                    masterVarIndices.Add(varIndex[mv]);
            }

            // We have only one constraint (LandAllocation) for the master in this example
            int numMasterCons = 1;
            int numMasterVars = masterVarIndices.Count;
            double[] masterCoefficients = new double[numMasterVars];
            double[,] masterLhs = new double[numMasterCons, numMasterVars];
            double[] masterRhs = new double[numMasterCons];

            for (int i = 0; i < numMasterCons; i++)
            {
                masterRhs[i] = rhs[i];
                for (int j = 0; j < numMasterVars; j++)
                {
                    masterLhs[i, j] = lhs[i, masterVarIndices[j]];
                }
            }
            for (int j = 0; j < numMasterVars; j++)
            {
                masterCoefficients[j] = objCoeffs[masterVarIndices[j]];
            }

            // In a real scenario, identify and build subproblem LHS/RHS from second-stage variables/constraints
            double[] subproblemCoefficients = new double[0];
            double[,] subproblemLhs = new double[0, 0];
            double[] subproblemRhs = new double[0];

            return (masterCoefficients, subproblemCoefficients, masterLhs, masterRhs, subproblemLhs, subproblemRhs, numMasterCons, numMasterVars, masterVars);
        }

        // Solve subproblem scenario given master solution and scenario multipliers.
        // In a real application, build and solve the subproblem LP, extract duals.
        static (bool feasible, double scenarioCost, double[] duals, bool farkas)
        SolveSubproblemScenario(double[] masterSol, string[] masterVarNames, double multiplier)
        {
            // Build subproblem constraints using masterSol (x-values) and scenario data
            // Solve subproblem using your LP solver
            // Extract dual multipliers from subproblem solution for cuts

            // Here, we assume feasibility and return a dummy scenarioCost and duals.
            // Replace with real subproblem solving logic.
            bool feasible = true;
            double scenarioCost = 1000 * multiplier; // Replace with actual computed scenario cost
            double[] duals = new double[] { 0.5 }; // Replace with actual dual values from subproblem
            bool farkas = false; // If infeasible, set to true and provide Farkas ray duals

            return (feasible, scenarioCost, duals, farkas);
        }

        // Add feasibility cut derived from duals
        // Replace with actual dual-based feasibility cut formula
        static void AddFeasibilityCut(ref double[,] lhs, ref double[] rhs, ref double[] objCoeffs, string[] varNames, double[] duals)
        {
            int oldCons = rhs.Length;
            int oldVars = objCoeffs.Length;
            double[,] newLhs = new double[oldCons + 1, oldVars];
            for (int i = 0; i < oldCons; i++)
                for (int j = 0; j < oldVars; j++)
                    newLhs[i, j] = lhs[i, j];

            // Example feasibility cut: sum(-x) >= -100
            // In a real scenario, use duals and problem structure to form correct cut.
            for (int j = 0; j < oldVars; j++)
            {
                if (varNames[j].StartsWith("x_"))
                    newLhs[oldCons, j] = -1.0;
                else
                    newLhs[oldCons, j] = 0.0;
            }

            double[] newRhs = new double[oldCons + 1];
            for (int i = 0; i < oldCons; i++) newRhs[i] = rhs[i];
            newRhs[oldCons] = -100.0; // Adjust based on dual computations

            lhs = newLhs;
            rhs = newRhs;
        }

        // Add optimality cut derived from duals
        // Replace with actual dual-based optimality cut formula
        static void AddOptimalityCut(ref double[,] lhs, ref double[] rhs, ref double[] objCoeffs, string[] varNames, double scenarioCost, double[] duals)
        {
            int oldCons = rhs.Length;
            int oldVars = objCoeffs.Length;
            double[,] newLhs = new double[oldCons + 1, oldVars];
            for (int i = 0; i < oldCons; i++)
                for (int j = 0; j < oldVars; j++)
                    newLhs[i, j] = lhs[i, j];

            // Example optimality cut:
            // theta >= scenarioCost - 50 -0.5*x_wheat -0.3*x_corn
            // In a real case, use duals to form these coefficients.
            for (int j = 0; j < oldVars; j++)
            {
                if (varNames[j] == "theta") newLhs[oldCons, j] = 1.0;
                else if (varNames[j] == "x_wheat") newLhs[oldCons, j] = -0.5;
                else if (varNames[j] == "x_corn") newLhs[oldCons, j] = -0.3;
                else newLhs[oldCons, j] = 0.0;
            }

            double[] newRhs = new double[oldCons + 1];
            for (int i = 0; i < oldCons; i++) newRhs[i] = rhs[i];
            newRhs[oldCons] = scenarioCost - 50.0; // Adjust based on dual computations

            lhs = newLhs;
            rhs = newRhs;
        }

        static double ExtractThetaValue(double[] sol, string[] varNames)
        {
            for (int i = 0; i < varNames.Length; i++)
            {
                if (varNames[i] == "theta") return sol[i];
            }
            return 0.0;
        }

        static void PrintSolution(string[] varNames, double[] sol)
        {
            if (sol == null) return;
            Console.WriteLine("Master Problem Final Solution:");
            for (int i = 0; i < varNames.Length; i++)
            {
                Console.WriteLine($"{varNames[i]} = {sol[i]:F4}");
            }
        }

        static (bool feasible, double[] solution, double objVal) SolveUsingPrimalSimplex(
    double[,] lhs, double[] rhs, double[] objectiveCoefficients, int totalVars, int numConstraints)
        {
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            bool shouldContinue = true;
            int iterationCount = 0;
            int originalVars = totalVars - numConstraints;

            double[] originalObjCoeffs = new double[totalVars];
            Array.Copy(objectiveCoefficients, originalObjCoeffs, totalVars);

            // Basic variables are the slack variables initially
            int[] basicVariables = new int[numConstraints];
            for (int i = 0; i < numConstraints; i++)
            {
                basicVariables[i] = originalVars + i;
            }

            bool feasible = true;
            double[] solution = new double[totalVars];
            double objVal = 0.0;

            while (shouldContinue)
            {
                iterationCount++;
                // Compute dual variables pi
                double[] pi = new double[numConstraints];
                for (int i = 0; i < numConstraints; i++)
                {
                    pi[i] = objectiveCoefficients[basicVariables[i]];
                }

                // Compute reduced costs
                double[] reducedCosts = new double[totalVars];
                for (int j = 0; j < totalVars; j++)
                {
                    reducedCosts[j] = objectiveCoefficients[j];
                    for (int i = 0; i < numConstraints; i++)
                    {
                        reducedCosts[j] -= pi[i] * lhs[i, j];
                    }
                }

                // Find entering variable
                int enteringVarIndex = -1;
                double maxRC = 0;
                for (int j = 0; j < totalVars; j++)
                {
                    if (reducedCosts[j] > maxRC)
                    {
                        maxRC = reducedCosts[j];
                        enteringVarIndex = j;
                    }
                }

                if (enteringVarIndex == -1)
                {
                    // No positive reduced cost, optimal solution found
                    shouldContinue = false;
                    break;
                }

                // Ratio test for leaving variable
                int leavingVarIndex = -1;
                double minRatio = double.MaxValue;
                for (int i = 0; i < numConstraints; i++)
                {
                    if (lhs[i, enteringVarIndex] > 0)
                    {
                        double ratio = rhs[i] / lhs[i, enteringVarIndex];
                        if (ratio < minRatio)
                        {
                            minRatio = ratio;
                            leavingVarIndex = i;
                        }
                    }
                }

                if (leavingVarIndex == -1)
                {
                    // Unbounded
                    shouldContinue = false;
                    feasible = false;
                    Console.WriteLine("Problem is unbounded.");
                    break;
                }

                // Pivot
                double pivot = lhs[leavingVarIndex, enteringVarIndex];
                for (int j = 0; j < totalVars; j++)
                {
                    lhs[leavingVarIndex, j] /= pivot;
                }
                rhs[leavingVarIndex] /= pivot;

                for (int i = 0; i < numConstraints; i++)
                {
                    if (i != leavingVarIndex)
                    {
                        double factor = lhs[i, enteringVarIndex];
                        for (int j = 0; j < totalVars; j++)
                        {
                            lhs[i, j] -= factor * lhs[leavingVarIndex, j];
                        }
                        rhs[i] -= factor * rhs[leavingVarIndex];
                    }
                }

                double objFactor = objectiveCoefficients[enteringVarIndex];
                for (int j = 0; j < totalVars; j++)
                {
                    objectiveCoefficients[j] -= objFactor * lhs[leavingVarIndex, j];
                }

                basicVariables[leavingVarIndex] = enteringVarIndex;
            }

            stopwatch.Stop();
            Console.WriteLine($"Primal simplex completed in {stopwatch.Elapsed.TotalSeconds} s");

            // Compute the final solution from the final tableau
            // Basic variables are in basicVariables[], their values in rhs[]
            // Non-basic variables are zero
            HashSet<int> basicSet = new HashSet<int>(basicVariables);
            for (int i = 0; i < numConstraints; i++)
            {
                solution[basicVariables[i]] = rhs[i];
            }

            // Compute objective value from the final tableau
            // After simplex ends, objectiveCoefficients array holds shifted values, but we can recompute:
            // If original objective is known, we can recompute from solution and original objective:
            // For demonstration, let's assume objectiveCoefficients[ ... ] now represent the final reduced costs and we can
            // sum up solution * originalObjCoeffs:
            for (int j = 0; j < totalVars; j++)
            {
                objVal += solution[j] * originalObjCoeffs[j];
            }

            return (feasible, solution, objVal);
        }

    }
}








/*
 //====================Some functions================================
 static (double[] masterCoefficients, double[] subproblemCoefficient,
 double[,] masterLhs, double[] masterRhs,
 double[,] subproblemLhs, double[] subproblemRhs,
 int masterConstraints, int masterTotalVars)
 ExtractMasterAndSubproblemData(
 List<string> allVariables, double[] objectiveCoefficients,
 double[,] lhsMatrix, double[] rhsMatrix, string[] masterVariables)
 {


     // Extract master LHS and RHS only for master variables
     var masterSelectedRows = new List<int>(); // Master 限制式行索引
     var subproblemSelectedRows = new List<int>(); // Subproblem 限制式行索引

     for (int i = 0; i < lhsMatrix.GetLength(0); i++) // 遍歷所有限制式行
     {
         bool isMasterRow = true;

         for (int j = 0; j < lhsMatrix.GetLength(1); j++) // 遍歷行中所有變數
         {
             // 如果 LHS 行中存在非零係數，但變數不屬於 Master Variables，則該行不屬於 Master Problem
             if (lhsMatrix[i, j] != 0 && !masterVariables.Contains(allVariables[j]))
             {
                 isMasterRow = false;
                 break;
             }
         }

         if (isMasterRow)
         {
             masterSelectedRows.Add(i);
         }
         else
         {
             subproblemSelectedRows.Add(i);
         }
     }

     // 構建 Master Problem 的 LHS 和 RHS
     // Master constraints and total variables
     int masterConstraints = masterSelectedRows.Count;
     int masterSlackVariables = masterConstraints;
     int masterTotalVars = lhsMatrix.GetLength(1) + masterSlackVariables;
     double[,] masterLhs = new double[masterConstraints, masterTotalVars];
     double[] masterRhs = new double[masterConstraints];

     for (int i = 0; i < masterSelectedRows.Count; i++)
     {
         int rowIndex = masterSelectedRows[i];
         for (int j = 0; j < lhsMatrix.GetLength(1); j++)
         {
             masterLhs[i, j] = lhsMatrix[rowIndex, j];
         }
         // 填充 Slack Variables (對角線)
         masterLhs[i, lhsMatrix.GetLength(1) + i] = 1;

         masterRhs[i] = rhsMatrix[rowIndex];
     }

     // 構建 Subproblem 的 LHS 和 RHS
     int subproblemConstraints = subproblemSelectedRows.Count;
     int subproblemSlackVariables = subproblemConstraints;

     double[,] subproblemLhs = new double[subproblemConstraints, lhsMatrix.GetLength(1) + subproblemSlackVariables];
     double[] subproblemRhs = new double[subproblemSelectedRows.Count];

     for (int i = 0; i < subproblemSelectedRows.Count; i++)
     {
         int rowIndex = subproblemSelectedRows[i];
         for (int j = 0; j < lhsMatrix.GetLength(1); j++)
         {
             subproblemLhs[i, j] = lhsMatrix[rowIndex, j];
         }
         // 填充 Slack Variables (對角線)
         subproblemLhs[i, lhsMatrix.GetLength(1) + i] = 1;
         subproblemRhs[i] = rhsMatrix[rowIndex];
     }

     var masterCoefficients = new double[objectiveCoefficients.Length + masterConstraints];
     var subproblemCoefficients = new double[objectiveCoefficients.Length + masterConstraints];
     var masterSet = new HashSet<string>(masterVariables.Select(s => s.Trim()));

     for (int i = 0; i < allVariables.Count; i++)
     {
         //Console.WriteLine($"Processing index {i}, allVariables[i]: '{allVariables[i]}', objectiveCoefficients[i]: {objectiveCoefficients[i]}");

         if (masterSet.Contains(allVariables[i]))
         {
             masterCoefficients[i] = objectiveCoefficients[i];
             subproblemCoefficients[i] = 0.0;
             Console.WriteLine($"Matched Master Variable: {allVariables[i]}");
         }
         else
         {
             subproblemCoefficients[i] = objectiveCoefficients[i];
             masterCoefficients[i] = 0.0;
             Console.WriteLine($"Matched Subproblem Variable: {allVariables[i]}");

         }
     }
     // Add Slack Variables with coefficients = 1
     for (int i = 0; i < masterConstraints; i++)
     {
         int slackVariableIndex = allVariables.Count + i; // Slack變數的索引位置
         if (slackVariableIndex < masterCoefficients.Length)
         {
             masterCoefficients[slackVariableIndex] = 1.0; // Slack 變數係數設為 1
             subproblemCoefficients[slackVariableIndex] = 1.0;
         }
     }

     return (masterCoefficients, subproblemCoefficients,
         masterLhs, masterRhs, subproblemLhs, subproblemRhs,
         masterConstraints, masterTotalVars);

 }


 static double[] BuildObjectiveCoefficients(List<string> variables, int numScenarios)
 {
     // Initialize coefficients based on variables
     Dictionary<string, double> allcoefficients = new Dictionary<string, double>
{ { "x_1", 150 }, { "x_2", 230 },{ "x_3", 260 }};

     for (int s = 1; s <= numScenarios; s++)
     {
         allcoefficients[$"w_1_{s}"] = -170.0 / numScenarios;
         allcoefficients[$"w_2_{s}"] = -150.0 / numScenarios;
         allcoefficients[$"w_3_{s}"] = -36.0 / numScenarios;
         allcoefficients[$"w_4_{s}"] = -10.0 / numScenarios;
         allcoefficients[$"y_1_{s}"] = 238.0 / numScenarios;
         allcoefficients[$"y_2_{s}"] = 210.0 / numScenarios;
     }

     // Map coefficients to the variable list
     double[] result = new double[variables.Count];
     for (int i = 0; i < variables.Count; i++)
     {
         result[i] = allcoefficients.ContainsKey(variables[i]) ? allcoefficients[variables[i]] : 0.0;
     }
     return result;
 }

 static List<string> GenerateVariables(int numScenarios)
 {
     // Generate variables based on the number of scenarios
     List<string> variables = new List<string>();

     // Master problem decision variables (e.g., x1, x2, x3)
     for (int i = 1; i <= 3; i++) // Assuming three x variables
     {
         variables.Add($"x_{i}");
     }
     // Subproblem variables (e.g., w1x, y1x, etc.)
     for (int s = 1; s <= numScenarios; s++)
     {
         variables.Add($"w_1_{s}");
         variables.Add($"w_2_{s}");
         variables.Add($"w_3_{s}");
         variables.Add($"w_4_{s}");
         variables.Add($"y_1_{s}");
         variables.Add($"y_2_{s}");
     }

     return variables;
 }

 static double[,] GenerateLhsMatrix(List<string> variables, int numScenarios)
 {
     int numConstraints = 1 + numScenarios * 3 + numScenarios; // 1 total land constraint + 3 constraints per scenario + 1 notdifferent constraint
     double[,] lhsMatrix = new double[numConstraints, variables.Count];

     // Total land constraint: x1 + x2 + x3 <= 500
     lhsMatrix[0, variables.IndexOf("x_1")] = 1;
     lhsMatrix[0, variables.IndexOf("x_2")] = 1;
     lhsMatrix[0, variables.IndexOf("x_3")] = 1;

     // Generate coefficients for each scenario
     for (int s = 1; s <= numScenarios; s++)
     {
         int row = 1 + (s - 1) * 3;

         // Generate uniformly distributed coefficients
         double[] x1Coeffs = GetUniformDistribution(2.5, 0.2, numScenarios);
         double[] x2Coeffs = GetUniformDistribution(3.0, 0.2, numScenarios);
         double[] x3Coeffs = GetUniformDistribution(20.0, 0.2, numScenarios);

         // Wheat constraint
         lhsMatrix[row, variables.IndexOf("x_1")] = -x1Coeffs[s - 1];
         lhsMatrix[row, variables.IndexOf($"y_1_{s}")] = -1;
         lhsMatrix[row, variables.IndexOf($"w_1_{s}")] = 1;

         // Corn constraint
         lhsMatrix[row + 1, variables.IndexOf("x_2")] = -x2Coeffs[s - 1];
         lhsMatrix[row + 1, variables.IndexOf($"y_2_{s}")] = -1;
         lhsMatrix[row + 1, variables.IndexOf($"w_2_{s}")] = 1;

         // Sugar constraint
         lhsMatrix[row + 2, variables.IndexOf("x_3")] = -x3Coeffs[s - 1];
         lhsMatrix[row + 2, variables.IndexOf($"w_3_{s}")] = 1;
         lhsMatrix[row + 2, variables.IndexOf($"w_4_{s}")] = 1;

         // w3 <= 6000 constraint
         int w3Row = 1 + numScenarios * 3 + (s - 1);
         lhsMatrix[w3Row, variables.IndexOf($"w_3_{s}")] = 1; // Coefficient of w3

     }

     return lhsMatrix;
 }

 static double[] GetUniformDistribution(double baseValue, double variability, int numScenarios)
 {
     // Generate uniformly distributed values across the range
     double minValue = baseValue * (1 - variability);
     double maxValue = baseValue * (1 + variability);
     double step = (maxValue - minValue) / (numScenarios - 1);

     double[] values = new double[numScenarios];
     for (int i = 0; i < numScenarios; i++)
     {
         values[i] = maxValue - i * step;
     }
     return values;
 }

 static double[] GenerateRhsMatrix(int numScenarios)
 {
     List<double> rhs = new List<double> { 500.0 }; // Total land constraint

     for (int s = 1; s <= numScenarios; s++)
     {
         rhs.Add(-200.0); // Wheat minimum constraint
         rhs.Add(-240.0); // Corn minimum constraint
         rhs.Add(0.0);    // Sugar constraint
     }

     for (int s = 1; s <= numScenarios; s++)
     {
         rhs.Add(6000.0); // w3 <= 6000 constraint for each scenario
     }

     return rhs.ToArray();
 }

 //============print the variable to check the mastercoeff, masterLhs,  masterRhs, subproblemCoeff===============
 // Display the objective coefficients
 Console.WriteLine("\nObjective Coefficient Matrix:");
 for (int i = 0; i < allVariables.Count; i++)
 {
     Console.WriteLine($"{allVariables[i]}: {objectiveCoefficients[i]}");
 }

 // Display the LHS matrix
 Console.WriteLine("\nLHS Matrix (Constraint Coefficients):");
 for (int i = 0; i < lhsMatrix.GetLength(0); i++)
 {
     for (int j = 0; j < lhsMatrix.GetLength(1); j++)
     {
         Console.Write($"{lhsMatrix[i, j],8:F2} ");
     }
     Console.WriteLine();
 }

 // Display the RHS matrix
 Console.WriteLine("\nRHS Matrix:");
 foreach (var rhs in rhsMatrix)
 {
     Console.WriteLine($"{rhs:F2}");
 }

 // Display Master Information
 Console.WriteLine("\nMaster Problem Coefficients:");
 for (int i = 0; i < masterCoefficients.GetLength(0); i++)
 {
     Console.Write($"{masterCoefficients[i],8:F2} ");
 }

 Console.WriteLine("\nMaster LHS Matrix:");
 for (int i = 0; i < masterLhs.GetLength(0); i++)
 {
     for (int j = 0; j < masterLhs.GetLength(1); j++)
     {
         Console.Write($"{masterLhs[i, j],8:F2} ");
     }
     Console.WriteLine();
 }

 Console.WriteLine("\nMaster RHS Matrix:");
 foreach (var rhs in masterRhs)
 {
     Console.WriteLine($"{rhs:F2}");
 }


 Console.WriteLine("\nSubproblem Coefficients:");
 for (int i = 0; i < subproblemCoefficients.GetLength(0); i++)
 {
     Console.Write($"{subproblemCoefficients[i],8:F2}");
 }


 Console.WriteLine("\nSubproblem LHS Matrix:");
 for (int i = 0; i < subproblemLhs.GetLength(0); i++)
 {
     for (int j = 0; j < subproblemLhs.GetLength(1); j++)
     {
         Console.Write($"{subproblemLhs[i, j],8:F2} ");
     }
     Console.WriteLine();
 }

 Console.WriteLine("\nSubproblem RHS Matrix:");
 foreach (var rhs in subproblemRhs)
 {
     Console.WriteLine($"{rhs:F2}");
 }

 // Solve Using Primal Simplex Method
 static void SolveUsingPrimalSimplex(double[,] lhs, double[] rhs,
                                 double[] objectiveCoefficients, int totalVars, int numConstraints)
 {
     Stopwatch stopwatch = new Stopwatch();
     stopwatch.Start();

     bool shouldContinue = true;
     int iterationCount = 0;
     int originalVars = totalVars - numConstraints;

     // Store the original objective coefficients
     double[] objCoeffs = new double[totalVars];
     Array.Copy(objectiveCoefficients, objCoeffs, totalVars);

     // Initialize basic variables (slack variables)
     int[] basicVariables = new int[numConstraints];
     for (int i = 0; i < numConstraints; i++)
     {
         basicVariables[i] = originalVars + i;
     }

     while (shouldContinue)
     {
         iterationCount++;
         Console.WriteLine($"\n--- Iteration {iterationCount} ---");

         // Calculate the dual variables (pi)
         double[] pi = new double[numConstraints];
         for (int i = 0; i < numConstraints; i++)
         {
             pi[i] = objectiveCoefficients[basicVariables[i]];
         }

         // Compute reduced costs for all variables
         double[] reducedCosts = new double[totalVars];
         for (int j = 0; j < totalVars; j++)
         {
             reducedCosts[j] = objectiveCoefficients[j];
             for (int i = 0; i < numConstraints; i++)
             {
                 reducedCosts[j] -= pi[i] * lhs[i, j];
             }
         }

         // Identify the entering variable (most positive reduced cost)
         int enteringVarIndex = -1;
         double maxReducedCost = 0;
         for (int j = 0; j < totalVars; j++)
         {
             if (reducedCosts[j] > maxReducedCost)
             {
                 maxReducedCost = reducedCosts[j];
                 enteringVarIndex = j;
             }
         }

         // If no entering variable is found, the solution is optimal
         if (enteringVarIndex == -1)
         {
             Console.WriteLine("Optimal solution found.");
             shouldContinue = false;
             break;
         }

         Console.WriteLine($"Chosen entering variable: {(enteringVarIndex < originalVars ? "x" : "s")}{enteringVarIndex + 1}");

         // Perform ratio test to select leaving variable
         int leavingVarIndex = -1;
         double minRatio = double.MaxValue;
         for (int i = 0; i < numConstraints; i++)
         {
             if (lhs[i, enteringVarIndex] > 0)
             {
                 double ratio = rhs[i] / lhs[i, enteringVarIndex];
                 if (ratio < minRatio)
                 {
                     minRatio = ratio;
                     leavingVarIndex = i;
                 }
             }
         }

         // If no leaving variable is found, the problem is unbounded
         if (leavingVarIndex == -1)
         {
             Console.WriteLine("Problem is unbounded.");
             shouldContinue = false;
             break;
         }

         Console.WriteLine($"Chosen leaving variable: {(basicVariables[leavingVarIndex] < originalVars ? "x" : "s")}{basicVariables[leavingVarIndex] + 1}");

         // Perform pivot operation
         double pivotElement = lhs[leavingVarIndex, enteringVarIndex];

         // Update the leaving row in-place
         for (int j = 0; j < totalVars; j++)
         {
             lhs[leavingVarIndex, j] /= pivotElement;
         }
         rhs[leavingVarIndex] /= pivotElement;

         // Update all other rows in-place
         for (int i = 0; i < numConstraints; i++)
         {
             if (i != leavingVarIndex)
             {
                 double factor = lhs[i, enteringVarIndex];
                 for (int j = 0; j < totalVars; j++)
                 {
                     lhs[i, j] -= factor * lhs[leavingVarIndex, j];
                 }
                 rhs[i] -= factor * rhs[leavingVarIndex];
             }
         }

         // Update the objective coefficients in-place
         double objectiveFactor = objectiveCoefficients[enteringVarIndex];
         for (int j = 0; j < totalVars; j++)
         {
             objectiveCoefficients[j] -= objectiveFactor * lhs[leavingVarIndex, j];
         }

         // Update the basic variable index in-place
         basicVariables[leavingVarIndex] = enteringVarIndex;

         // Print the current tableau
         //PrintCurrentDictionaryForPrimal(objectiveCoefficients, lhs, rhs, basicVariables, totalVars, numConstraints);
     }


 }

}

}
}

/*










