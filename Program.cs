using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace StochasticFarmerProblem
{
    // Class to store variable info
    class Variable
    {
        public string Name { get; set; }
        public double Value { get; set; }
        public Variable(string name) { Name = name; Value = 0.0; }
    }

    class BendersDecomposition
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Enter the number of scenarios:");
            int numScenarios;
            while (!int.TryParse(Console.ReadLine(), out numScenarios) || numScenarios <= 0)
            {
                Console.WriteLine("Invalid input. Enter positive integer for scenarios:");
            }

            // Generate all variables (x_1,x_2,x_3 for first-stage, y_1_s,y_2_s,w_1_s,... for second-stage)
            List<string> allVariables = GenerateAllVariables(numScenarios);

            Console.WriteLine("\nAll Variables:");
            foreach (var v in allVariables) Console.Write(v + " ");
            Console.WriteLine();

            Console.WriteLine("\nEnter variables to include in master problem (comma-separated):");
            string inputVars = Console.ReadLine();
            string[] masterVars = inputVars.Split(',').Select(s => s.Trim()).Where(s => s != "").ToArray();

            // Build full objective and constraints
            double[] fullObj = BuildObjectiveCoefficients(allVariables, numScenarios);
            double[,] fullLHS = BuildFullLHS(allVariables, numScenarios);
            double[] fullRHS = BuildFullRHS(numScenarios);

            // Extract master data
            var masterData = ExtractMasterData(allVariables, fullObj, fullLHS, fullRHS, masterVars);
            double[] masterCoefficients = masterData.masterCoefficients;
            double[,] masterLHS = masterData.masterLHS;
            double[] masterRHS = masterData.masterRhs;
            int masterCons = masterData.masterCons;
            int masterTotalVars = masterData.masterTotalVars;
            string[] masterVarNames = masterData.masterVarNames;
            List<int> masterVarIndices = masterData.masterVarIndices;

            // Check complete recourse
            bool isCompleteRecourse = CheckCompleteRecourse(masterVars);

            // Scenario multipliers
            double[] scenarioMultipliers = ComputeScenarioMultipliers(numScenarios);

            double UB = double.PositiveInfinity;
            double LB = double.NegativeInfinity;
            double tolerance = 1e-6;
            int maxIterations = 50;
            int scenarioIndex = 0;

            for (int iteration = 1; iteration <= maxIterations; iteration++)
            {
                Console.WriteLine($"\n--- Benders Iteration {iteration} ---");

                var masterRes = SolveUsingPrimalSimplex(masterLHS, masterRHS, masterCoefficients, masterTotalVars, masterCons);
                if (!masterRes.feasible)
                {
                    Console.WriteLine("Master infeasible. Stopping.");
                    break;
                }

                LB = masterRes.objVal;
                double[] masterSol = masterRes.solution;

                // Choose scenario
                int s = scenarioIndex;
                scenarioIndex = (scenarioIndex + 1) % numScenarios;

                // Extract x_1,x_2,x_3,theta
                double x1 = 0, x2 = 0, x3 = 0, thetaVal = 0;
                for (int i = 0; i < masterVarNames.Length; i++)
                {
                    if (masterVarNames[i] == "x_1") x1 = masterSol[i];
                    else if (masterVarNames[i] == "x_2") x2 = masterSol[i];
                    else if (masterVarNames[i] == "x_3") x3 = masterSol[i];
                    else if (masterVarNames[i] == "theta") thetaVal = masterSol[i];
                }

                var subRes = SolveSubproblemScenario(x1, x2, x3, scenarioMultipliers[s], s + 1, numScenarios, masterVars);
                bool feasibleSP = subRes.feasible;
                double scenarioCost = subRes.objVal;
                double[] subDuals = subRes.duals;
                bool farkas = subRes.farkas;

                bool anyCutAdded = false;
                if (!feasibleSP && !isCompleteRecourse)
                {
                    AddFeasibilityCut(ref masterLHS, ref masterRHS, ref masterCoefficients, masterVarNames, subDuals);
                    anyCutAdded = true;
                    Console.WriteLine("Feasibility cut added.");
                }
                else if (feasibleSP)
                {
                    AddOptimalityCut(ref masterLHS, ref masterRHS, ref masterCoefficients, masterVarNames, scenarioCost, subDuals);
                    anyCutAdded = true;
                    Console.WriteLine("Optimality cut added.");
                    UB = Math.Min(UB, LB + (scenarioCost - thetaVal));
                }

                if (!anyCutAdded && Math.Abs(UB - LB) < tolerance)
                {
                    Console.WriteLine("Converged: UB ~ LB");
                    break;
                }
            }

            var finalRes = SolveUsingPrimalSimplex(masterLHS, masterRHS, masterCoefficients, masterTotalVars, masterCons);
            PrintFinalSolution(masterVarNames, finalRes.solution);
        }

        static List<string> GenerateAllVariables(int numScenarios)
        {
            List<string> vars = new List<string>();
            vars.Add("theta");
            vars.Add("x_1");
            vars.Add("x_2");
            vars.Add("x_3");

            for (int s = 1; s <= numScenarios; s++)
            {
                vars.Add($"y_1_{s}");
                vars.Add($"y_2_{s}");
            }
            for (int s = 1; s <= numScenarios; s++)
            {
                vars.Add($"w_1_{s}");
                vars.Add($"w_2_{s}");
                vars.Add($"w_3_{s}");
                vars.Add($"w_4_{s}");
            }
            return vars;
        }

        static double[] BuildObjectiveCoefficients(List<string> vars, int numScenarios)
        {
            double[] coeff = new double[vars.Count];
            for (int i = 0; i < vars.Count; i++)
            {
                string v = vars[i];
                if (v == "theta") coeff[i] = 1.0;
                else if (v == "x_1") coeff[i] = -150;
                else if (v == "x_2") coeff[i] = -230;
                else if (v == "x_3") coeff[i] = -260;
                else if (v.StartsWith("y_1_")) coeff[i] = 238.0 / numScenarios;
                else if (v.StartsWith("y_2_")) coeff[i] = 210.0 / numScenarios;
                else if (v.StartsWith("w_1_")) coeff[i] = -170.0 / numScenarios;
                else if (v.StartsWith("w_2_")) coeff[i] = -150.0 / numScenarios;
                else if (v.StartsWith("w_3_")) coeff[i] = -36.0 / numScenarios;
                else if (v.StartsWith("w_4_")) coeff[i] = -10.0 / numScenarios;
                else coeff[i] = 0.0;
            }
            return coeff;
        }

        static double[,] BuildFullLHS(List<string> vars, int numScenarios)
        {
            int cons = 1 + numScenarios * 4;
            int vcount = vars.Count;
            double[,] lhs = new double[cons, vcount];

            lhs[0, vars.IndexOf("x_1")] = 1;
            lhs[0, vars.IndexOf("x_2")] = 1;
            lhs[0, vars.IndexOf("x_3")] = 1;

            for (int s = 1; s <= numScenarios; s++)
            {
                int baseRow = 1 + (s - 1) * 4;
                lhs[baseRow, vars.IndexOf("x_1")] = -3.0;
                lhs[baseRow, vars.IndexOf($"y_1_{s}")] = -1.0;
                lhs[baseRow, vars.IndexOf($"w_1_{s}")] = 1.0;

                lhs[baseRow + 1, vars.IndexOf("x_2")] = -3.6;
                lhs[baseRow + 1, vars.IndexOf($"y_2_{s}")] = -1.0;
                lhs[baseRow + 1, vars.IndexOf($"w_2_{s}")] = 1.0;

                lhs[baseRow + 2, vars.IndexOf($"w_3_{s}")] = 1.0;
                lhs[baseRow + 2, vars.IndexOf($"w_4_{s}")] = 1.0;
                lhs[baseRow + 2, vars.IndexOf("x_3")] = -20.0;

                lhs[baseRow + 3, vars.IndexOf($"w_3_{s}")] = 1.0;
            }

            return lhs;
        }

        static double[] BuildFullRHS(int numScenarios)
        {
            List<double> rhs = new List<double>();
            rhs.Add(500.0); // land
            for (int s = 1; s <= numScenarios; s++)
            {
                rhs.Add(-200.0); // wheat
                rhs.Add(-240.0); // corn
                rhs.Add(0.0);    // sugar
                rhs.Add(6000.0); // w_3_s ≤6000
            }
            return rhs.ToArray();
        }

        static (double[] masterCoefficients, double[] subproblemCoefficients,
                double[,] masterLHS, double[] masterRhs,
                int masterCons, int masterTotalVars,
                string[] masterVarNames, List<int> masterVarIndices)
        ExtractMasterData(
            List<string> allVars, double[] objCoeffs, double[,] lhs, double[] rhs, string[] masterVars)
        {
            int numCons = lhs.GetLength(0);
            int numVars = lhs.GetLength(1);

            Dictionary<string, int> vIndex = new Dictionary<string, int>();
            for (int i = 0; i < allVars.Count; i++) vIndex[allVars[i]] = i;

            List<int> mIndices = new List<int>();
            HashSet<string> mSet = new HashSet<string>(masterVars);
            for (int i = 0; i < allVars.Count; i++)
            {
                if (mSet.Contains(allVars[i]))
                    mIndices.Add(i);
            }

            int masterCons = numCons;
            int masterTotalVars = mIndices.Count + masterCons;

            double[,] masterLhs = new double[masterCons, masterTotalVars];
            double[] masterRhs = new double[masterCons];
            for (int i = 0; i < masterCons; i++)
            {
                for (int j = 0; j < mIndices.Count; j++)
                    masterLhs[i, j] = lhs[i, mIndices[j]];
                masterLhs[i, mIndices.Count + i] = 1.0; // slack
                masterRhs[i] = rhs[i];
            }

            double[] masterCoefficients = new double[masterTotalVars];
            double[] subproblemCoefficients = new double[masterTotalVars];
            for (int j = 0; j < mIndices.Count; j++)
                masterCoefficients[j] = objCoeffs[mIndices[j]];
            for (int i = 0; i < masterCons; i++)
                masterCoefficients[mIndices.Count + i] = 0.0; // slack cost=0

            string[] masterVarNames = new string[masterTotalVars];
            for (int j = 0; j < mIndices.Count; j++)
                masterVarNames[j] = allVars[mIndices[j]];
            for (int i = 0; i < masterCons; i++)
                masterVarNames[mIndices.Count + i] = $"slack_{i + 1}";

            return (masterCoefficients, subproblemCoefficients, masterLhs, masterRhs,
                    masterCons, masterTotalVars, masterVarNames, mIndices);
        }

        static bool CheckCompleteRecourse(string[] masterVars)
        {
            foreach (var mv in masterVars)
            {
                if (mv.StartsWith("y_") || mv.StartsWith("w_"))
                    return false;
            }
            return true;
        }

        static double[] ComputeScenarioMultipliers(int numScenarios)
        {
            double[] m = new double[numScenarios];
            if (numScenarios == 1) { m[0] = 1.0; return m; }
            for (int i = 0; i < numScenarios; i++)
                m[i] = 0.8 + 0.4 * i / (numScenarios - 1);
            return m;
        }

        static (bool feasible, double objVal, double[] duals, bool farkas)
        SolveSubproblemScenario(double x1, double x2, double x3, double multiplier, int scenarioIndex, int numScenarios, string[] masterVars)
        {
            double wheatProd = x1 * 2.5 * multiplier;
            double cornProd = x2 * 3.0 * multiplier;
            double sugarProd = x3 * 20.0 * multiplier;

            List<string> subVars = new List<string> { "y_1", "y_2", "w_1", "w_2", "w_3", "w_4" };
            int svCount = subVars.Count;

            double[] subObj = new double[svCount];
            subObj[subVars.IndexOf("y_1")] = 238.0;
            subObj[subVars.IndexOf("y_2")] = 210.0;
            subObj[subVars.IndexOf("w_1")] = -170.0;
            subObj[subVars.IndexOf("w_2")] = -150.0;
            subObj[subVars.IndexOf("w_3")] = -36.0;
            subObj[subVars.IndexOf("w_4")] = -10.0;

            int scCons = 4;
            double[,] scLhs = new double[scCons, svCount];
            double[] scRhs = new double[scCons];

            // y_1-w_1 ≤ wheatProd-200
            scLhs[0, subVars.IndexOf("y_1")] = 1.0;
            scLhs[0, subVars.IndexOf("w_1")] = -1.0;
            scRhs[0] = wheatProd - 200;

            // y_2-w_2 ≤ cornProd-240
            scLhs[1, subVars.IndexOf("y_2")] = 1.0;
            scLhs[1, subVars.IndexOf("w_2")] = -1.0;
            scRhs[1] = cornProd - 240;

            // w_3+w_4 ≤ sugarProd
            scLhs[2, subVars.IndexOf("w_3")] = 1.0;
            scLhs[2, subVars.IndexOf("w_4")] = 1.0;
            scRhs[2] = sugarProd;

            // w_3 ≤6000
            scLhs[3, subVars.IndexOf("w_3")] = 1.0;
            scRhs[3] = 6000.0;

            var res = SolveUsingPrimalSimplexSubproblem(scLhs, scRhs, subObj, svCount, scCons);
            return res;
        }

        static void AddFeasibilityCut(ref double[,] lhs, ref double[] rhs, ref double[] objCoeffs, string[] varNames, double[] duals)
        {
            int oldCons = rhs.Length;
            int oldVars = objCoeffs.Length;
            double[,] newLhs = new double[oldCons + 1, oldVars];
            for (int i = 0; i < oldCons; i++)
                for (int j = 0; j < oldVars; j++)
                    newLhs[i, j] = lhs[i, j];

            for (int j = 0; j < oldVars; j++)
            {
                if (varNames[j] == "x_1" || varNames[j] == "x_2" || varNames[j] == "x_3")
                    newLhs[oldCons, j] = -1.0;
                else
                    newLhs[oldCons, j] = 0.0;
            }

            double[] newRhs = new double[oldCons + 1];
            for (int i = 0; i < oldCons; i++) newRhs[i] = rhs[i];
            newRhs[oldCons] = -100.0;

            lhs = newLhs;
            rhs = newRhs;
        }

        static void AddOptimalityCut(ref double[,] lhs, ref double[] rhs, ref double[] objCoeffs, string[] varNames, double scenarioCost, double[] duals)
        {
            int oldCons = rhs.Length;
            int oldVars = objCoeffs.Length;
            double[,] newLhs = new double[oldCons + 1, oldVars];
            for (int i = 0; i < oldCons; i++)
                for (int j = 0; j < oldVars; j++)
                    newLhs[i, j] = lhs[i, j];

            for (int j = 0; j < oldVars; j++)
            {
                if (varNames[j] == "theta") newLhs[oldCons, j] = 1.0;
                else if (varNames[j] == "x_1") newLhs[oldCons, j] = -0.5;
                else if (varNames[j] == "x_2") newLhs[oldCons, j] = -0.3;
                else newLhs[oldCons, j] = 0.0;
            }

            double[] newRhs = new double[oldCons + 1];
            for (int i = 0; i < oldCons; i++) newRhs[i] = rhs[i];
            newRhs[oldCons] = scenarioCost - 50.0;

            lhs = newLhs;
            rhs = newRhs;
        }

        static void PrintFinalSolution(string[] varNames, double[] sol)
        {
            if (sol == null) return;

            List<Variable> vars = new List<Variable>();
            for (int i = 0; i < varNames.Length; i++)
            {
                vars.Add(new Variable(varNames[i]) { Value = sol[i] });
            }

            Console.WriteLine("\nMaster Problem Final Optimal Solution:");
            foreach (var v in vars)
            {
                Console.WriteLine($"{v.Name} = {v.Value:F4}");
            }
        }

        static (bool feasible, double[] solution, double objVal) SolveUsingPrimalSimplex(
    double[,] lhs, double[] rhs, double[] objCoeffs, int totalVars, int numCons)
        {
            // Implement full primal simplex
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            bool shouldContinue = true;
            int iterationCount = 0;
            int originalVars = totalVars - numCons;

            // Store the original objective coefficients
            double[] originalObjCoeffs = new double[totalVars];
            Array.Copy(objCoeffs, originalObjCoeffs, totalVars);

            // Initialize basic variables (slack variables)
            int[] basicVariables = new int[numCons];
            for (int i = 0; i < numCons; i++)
            {
                basicVariables[i] = originalVars + i;
            }

            while (shouldContinue)
            {
                iterationCount++;
                Console.WriteLine($"\n--- Iteration {iterationCount} ---");

                // Calculate the dual variables (pi)
                double[] pi = new double[numCons];
                for (int i = 0; i < numCons; i++)
                {
                    pi[i] = objCoeffs[basicVariables[i]];
                }

                // Compute reduced costs for all variables
                double[] reducedCosts = new double[totalVars];
                for (int j = 0; j < totalVars; j++)
                {
                    reducedCosts[j] = objCoeffs[j];
                    for (int i = 0; i < numCons; i++)
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
                for (int i = 0; i < numCons; i++)
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
                    return (false, null, double.PositiveInfinity);
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
                for (int i = 0; i < numCons; i++)
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
                double objectiveFactor = objCoeffs[enteringVarIndex];
                for (int j = 0; j < totalVars; j++)
                {
                    objCoeffs[j] -= objectiveFactor * lhs[leavingVarIndex, j];
                }

                // Update the basic variable index in-place
                basicVariables[leavingVarIndex] = enteringVarIndex;
            }

            // Extract solution
            bool feasible = true;
            double[] solution = new double[totalVars];
            for (int i = 0; i < numCons; i++)
            {
                solution[basicVariables[i]] = rhs[i];
            }

            // Calculate the final objective value
            double objVal = 0.0;
            for (int i = 0; i < originalVars; i++)
            {
                objVal += solution[i] * originalObjCoeffs[i];
            }

            stopwatch.Stop();
            Console.WriteLine($"Optimal solution found in {iterationCount} iterations.");
            Console.WriteLine($"Time taken: {stopwatch.ElapsedMilliseconds} ms");

            return (feasible, solution, objVal);
        }


        static (bool feasible, double objVal, double[] duals, bool farkas)
            SolveUsingPrimalSimplexSubproblem(double[,] lhs, double[] rhs, double[] objCoeffs, int totalVars, int numCons)
        {
            // Implement primal simplex for subproblem and dual/farkas extraction
            // Return feasible solution for now
            bool feasible = true;
            double objVal = 1000.0;
            double[] duals = new double[numCons];
            bool farkas = false;
            return (feasible, objVal, duals, farkas);
        }
    }
}
