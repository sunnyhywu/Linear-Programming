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
            List<string> allVariables = GenerateVariables(numScenarios);

            // Display the list of available variables to the user
            Console.WriteLine($"\nAvailable variables for {numScenarios} scenarios:");
            foreach (var variable in allVariables)
            {
                Console.Write(variable + " ");
            }
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
            var (masterCoefficients, subproblemCoefficients, masterLhs, masterRhs, subproblemLhs, subproblemRhs) =
                ExtractMasterAndSubproblemData(
                    allVariables,
                    objectiveCoefficients,
                    lhsMatrix,
                    rhsMatrix,
                    masterVariables);

                static (double[,] masterCoefficientsMatrix, double[,] subproblemCoefficientsMatrix, double[,] masterLhs, double[] masterRhs, double[,] subproblemLhs, double[] subproblemRhs)
                ExtractMasterAndSubproblemData(
                 List<string> allVariables, double[] objectiveCoefficients, double[,] lhsMatrix, double[] rhsMatrix, string[] masterVariables)
                  {
                var masterCoefficientsMatrix = new double[objectiveCoefficients.Length, 1];
                var subproblemCoefficientsMatrix = new double[objectiveCoefficients.Length, 1];
                var masterSet = new HashSet<string>(masterVariables.Select(s => s.Trim()));

                for (int i = 0; i < allVariables.Count; i++)
                {
                    //Console.WriteLine($"Processing index {i}, allVariables[i]: '{allVariables[i]}', objectiveCoefficients[i]: {objectiveCoefficients[i]}");

                    if (masterSet.Contains(allVariables[i]))
                    {
                        masterCoefficientsMatrix[i, 0] = objectiveCoefficients[i];
                        subproblemCoefficientsMatrix[i, 0] = 0.0;
                        Console.WriteLine($"Matched Master Variable: {allVariables[i]}");
                    }
                    else
                    {
                        subproblemCoefficientsMatrix[i, 0] = objectiveCoefficients[i];
                        masterCoefficientsMatrix[i, 0] = 0.0;
                        Console.WriteLine($"Matched Subproblem Variable: {allVariables[i]}");

                    }
                }

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
                double[,] masterLhs = new double[masterSelectedRows.Count, lhsMatrix.GetLength(1)];
                double[] masterRhs = new double[masterSelectedRows.Count];

                for (int i = 0; i < masterSelectedRows.Count; i++)
                {
                    int rowIndex = masterSelectedRows[i];
                    for (int j = 0; j < lhsMatrix.GetLength(1); j++)
                    {
                        masterLhs[i, j] = lhsMatrix[rowIndex, j];
                    }
                    masterRhs[i] = rhsMatrix[rowIndex];
                }

                // 構建 Subproblem 的 LHS 和 RHS
                double[,] subproblemLhs = new double[subproblemSelectedRows.Count, lhsMatrix.GetLength(1)];
                double[] subproblemRhs = new double[subproblemSelectedRows.Count];

                for (int i = 0; i < subproblemSelectedRows.Count; i++)
                {
                    int rowIndex = subproblemSelectedRows[i];
                    for (int j = 0; j < lhsMatrix.GetLength(1); j++)
                    {
                        subproblemLhs[i, j] = lhsMatrix[rowIndex, j];
                    }
                    subproblemRhs[i] = rhsMatrix[rowIndex];
                }

                return (masterCoefficientsMatrix, subproblemCoefficientsMatrix, masterLhs, masterRhs, subproblemLhs, subproblemRhs);
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


            //============print  to check the mastercoeff, masterLhs,  masterRhs, subproblemCoeff===============
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
                Console.Write($"{masterCoefficients[i, 0],8:F2} ");
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
                Console.Write($"{subproblemCoefficients[i,0],8:F2}");
            }


            Console.WriteLine("\nSubproblem LHS Matrix:");
            for (int i = 0; i < subproblemLhs.GetLength(0); i++)
            {
                for (int j = 0; j < subproblemLhs.GetLength(1); j++)
                {
                    Console.Write($"{subproblemLhs[i,j],8:F2} ");
                }
                Console.WriteLine();
            }
            
            Console.WriteLine("\nSubproblem RHS Matrix:");
            foreach (var rhs in subproblemRhs)
            {
                Console.WriteLine($"{rhs:F2}");
            }

        }

    }

}

















/*
namespace StochasticFarmerProblem
{
    class Program
    {
        static void Main(string[] args)
        {
            // Define scenario counts to evaluate
            int[] scenarioCounts = { 100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000 };

            // Problem description
            double totalLand = 500;
            Dictionary<string, double> yields = new Dictionary<string, double> { { "wheat", 2.5 }, { "corn", 3.0 }, { "sugar", 20.0 } };
            Dictionary<string, double> plantingCosts = new Dictionary<string, double> { { "wheat", 150 }, { "corn", 230 }, { "sugar", 260 } };
            Dictionary<string, double> sellingPrices = new Dictionary<string, double> { { "wheat", 170 }, { "corn", 150 }, { "sugar_favorable", 36 }, { "sugar_unfavorable", 10 } };
            Dictionary<string, double> purchasePrices = new Dictionary<string, double> { { "wheat", 238 }, { "corn", 210 } };
            Dictionary<string, double> minRequirements = new Dictionary<string, double> { { "wheat", 200 }, { "corn", 240 } };

            // Lists to store results for summary
            List<int> scenarioCountsList = new List<int>();
            List<double> runTimes = new List<double>();
            List<double> firstStageObjectiveValues = new List<double>();
            List<double> secondStageObjectiveValues = new List<double>();
            List<double> totalObjectiveValues = new List<double>();
            List<double> overallProfits = new List<double>();
            List<double> xWheatValues = new List<double>();
            List<double> xCornValues = new List<double>();
            List<double> xSugarValues = new List<double>();

            foreach (var scenarioCount in scenarioCounts)
            {
                Cplex model = new Cplex();

                // First-stage decision variables (land allocation)
                INumVar x_wheat = model.NumVar(0, totalLand, "x_wheat");
                INumVar x_corn = model.NumVar(0, totalLand, "x_corn");
                INumVar x_sugar = model.NumVar(0, totalLand, "x_sugar");

                // Land allocation constraint
                model.AddLe(model.Sum(x_wheat, x_corn, x_sugar), totalLand, "LandAllocation");

                // First-stage planting costs
                ILinearNumExpr firstStageCosts = model.LinearNumExpr();
                firstStageCosts.AddTerm(plantingCosts["wheat"], x_wheat);
                firstStageCosts.AddTerm(plantingCosts["corn"], x_corn);
                firstStageCosts.AddTerm(plantingCosts["sugar"], x_sugar);

                // Second-stage variables and constraints
                INumVar[] recourseVars = new INumVar[scenarioCount];
                INumVar[] wheat_purchased = new INumVar[scenarioCount];
                INumVar[] corn_purchased = new INumVar[scenarioCount];
                INumVar[] sugar_sold_at_36 = new INumVar[scenarioCount];
                INumVar[] sugar_sold_at_10 = new INumVar[scenarioCount];

                Random random = new Random();

                for (int s = 0; s < scenarioCount; s++)
                {
                    // Generate random yield multipliers
                    double yieldMultiplier_wheat = 1 + 0.2 * (random.NextDouble() - 0.5);
                    double yieldMultiplier_corn = 1 + 0.2 * (random.NextDouble() - 0.5);
                    double yieldMultiplier_sugar = 1 + 0.2 * (random.NextDouble() - 0.5);

                    // Second-stage variables for scenario s
                    wheat_purchased[s] = model.NumVar(0, double.MaxValue, $"wheat_purchased_{s}");
                    corn_purchased[s] = model.NumVar(0, double.MaxValue, $"corn_purchased_{s}");
                    sugar_sold_at_36[s] = model.NumVar(0, 6000, $"sugar_sold_at_36_{s}");
                    sugar_sold_at_10[s] = model.NumVar(0, double.MaxValue, $"sugar_sold_at_10_{s}");
                    recourseVars[s] = model.NumVar(double.MinValue, double.MaxValue, $"profit_{s}");

                    // Production expressions
                    INumExpr wheat_production = model.Prod(x_wheat, yields["wheat"] * yieldMultiplier_wheat);
                    INumExpr corn_production = model.Prod(x_corn, yields["corn"] * yieldMultiplier_corn);
                    INumExpr sugar_production = model.Prod(x_sugar, yields["sugar"] * yieldMultiplier_sugar);

                    // Constraints for minimum requirements
                    model.AddGe(model.Sum(wheat_production, wheat_purchased[s]), minRequirements["wheat"], $"WheatRequirement_{s}");
                    model.AddGe(model.Sum(corn_production, corn_purchased[s]), minRequirements["corn"], $"CornRequirement_{s}");

                    // Constraints for sugar beet sales
                    model.AddEq(model.Sum(sugar_sold_at_36[s], sugar_sold_at_10[s]), sugar_production, $"SugarSales_{s}");

                    // Revenue expressions
                    INumExpr wheat_revenue = model.Prod(sellingPrices["wheat"], wheat_production);
                    INumExpr corn_revenue = model.Prod(sellingPrices["corn"], corn_production);
                    INumExpr sugar_revenue = model.Sum(
                        model.Prod(sellingPrices["sugar_favorable"], sugar_sold_at_36[s]),
                        model.Prod(sellingPrices["sugar_unfavorable"], sugar_sold_at_10[s])
                    );

                    // Purchase cost expressions
                    INumExpr wheat_purchase_cost = model.Prod(purchasePrices["wheat"], wheat_purchased[s]);
                    INumExpr corn_purchase_cost = model.Prod(purchasePrices["corn"], corn_purchased[s]);

                    // Profit per scenario
                    INumExpr profit_s = model.Diff(
                        model.Sum(wheat_revenue, corn_revenue, sugar_revenue),
                        model.Sum(wheat_purchase_cost, corn_purchase_cost)
                    );

                    // Set recourseVars[s] to the profit per scenario
                    model.AddEq(recourseVars[s], profit_s, $"Recourse_{s}");
                }

                // Objective: Maximize expected profit (average over scenarios) minus first-stage costs
                ILinearNumExpr expectedProfit = model.LinearNumExpr();
                for (int s = 0; s < scenarioCount; s++)
                {
                    expectedProfit.AddTerm(1.0 / scenarioCount, recourseVars[s]);
                }

                // Total objective: Expected profit minus first-stage costs
                IObjective objective = model.AddMaximize(model.Diff(expectedProfit, firstStageCosts));

                // Start timer and solve the model
                Stopwatch stopwatch = new Stopwatch();
                stopwatch.Start();

                if (model.Solve())
                {
                    stopwatch.Stop();

                    // Retrieve objective values
                    double totalProfit = model.ObjValue;
                    double firstStageCostValue = model.GetValue(firstStageCosts);
                    double expectedSecondStageProfit = totalProfit + firstStageCostValue;
                    double overallProfit = expectedSecondStageProfit - firstStageCostValue;

                    // Retrieve first-stage decisions
                    double xWheat = model.GetValue(x_wheat);
                    double xCorn = model.GetValue(x_corn);
                    double xSugar = model.GetValue(x_sugar);

                    // Store values for summary and plotting
                    scenarioCountsList.Add(scenarioCount);
                    runTimes.Add(stopwatch.Elapsed.TotalSeconds);
                    firstStageObjectiveValues.Add(-firstStageCostValue);
                    secondStageObjectiveValues.Add(expectedSecondStageProfit);
                    totalObjectiveValues.Add(totalProfit);
                    overallProfits.Add(overallProfit);
                    xWheatValues.Add(xWheat);
                    xCornValues.Add(xCorn);
                    xSugarValues.Add(xSugar);
                }
                else
                {
                    Console.WriteLine($"No solution found for scenario count {scenarioCount}");
                }

                model.End();
            }

            // Define the EV solution values based on the provided EV table
            double evWheat = 120; // Acres for wheat in EV solution
            double evCorn = 80;   // Acres for corn in EV solution
            double evSugar = 300; // Acres for sugar beets in EV solution
            double evFirstStageCost = evWheat * plantingCosts["wheat"] + evCorn * plantingCosts["corn"] + evSugar * plantingCosts["sugar"];

            // Calculate the Stage 2 Objective (Second-Stage Profit) for the EV solution
            double evWheatSales = (300 - minRequirements["wheat"]) * sellingPrices["wheat"]; // Selling 100 tons of wheat
            double evCornSales = 0; // All corn is used to meet requirements, so no sales
            double evSugarSales = 6000 * sellingPrices["sugar_favorable"]; // All 6000 tons at favorable price

            double evSecondStageProfit = evWheatSales + evCornSales + evSugarSales; // 17000 + 0 + 216000 = 233000
            double evTotalProfit = evSecondStageProfit - evFirstStageCost;

            // Print the summary table with an additional row for EV solution
            Console.WriteLine("\nSummary:");
            Console.WriteLine("Scenario Count | Runtime (s) | First-Stage Obj | Second-Stage Obj | Overall Profit | Wheat Acres | Corn Acres | Sugar Acres");
            for (int i = 0; i < scenarioCountsList.Count; i++)
            {
                Console.WriteLine($"{scenarioCountsList[i],-14} | {runTimes[i]:F2}       | {firstStageObjectiveValues[i],-16:F2} | {secondStageObjectiveValues[i],-17:F2} | {overallProfits[i],-14:F2} | {xWheatValues[i],-11:F2} | {xCornValues[i],-10:F2} | {xSugarValues[i]:F2}");
            }
            // Print the EV solution row for easy comparison
            Console.WriteLine("Profit for EV Solution: ");
            Console.WriteLine($"EV Solution    | -          | {-evFirstStageCost:F2}       | {evSecondStageProfit:F2}         | {evTotalProfit:F2}      | {evWheat:F2}      | {evCorn:F2}      | {evSugar:F2}");
        
    }
    }
}
*/