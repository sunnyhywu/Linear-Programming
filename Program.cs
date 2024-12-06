using System;
using System.Collections.Generic;
using System.Diagnostics;
//using ILOG.CPLEX;
//using ILOG.Concert;

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
