using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

class ResourceAllocationProblem
{
    static double[] objCoeffs;

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

        // Output the final solution with shadow prices and slack variables
        OutputFinalSolutionWithShadowPricesAndSlacks(objCoeffs, lhs, rhs, basicVariables, totalVars, numConstraints);
       

        stopwatch.Stop();
        Console.WriteLine($"Total runtime for Primal method: {stopwatch.Elapsed.TotalSeconds} seconds");
    }


    // Shared output function to display the final solution, shadow prices, and slack variables
    static void OutputFinalSolutionWithShadowPricesAndSlacks(double[] objCoeffs, double[,] lhs, double[] rhs, int[] basicVariables, int totalVars, int numConstraints)
    {
        int originalVars = totalVars - numConstraints;
        Console.WriteLine("\nFinal Solution:");
        double optimalValue = 0;
        HashSet<int> basicSet = new HashSet<int>(basicVariables);

        // Display values for basic variables
        for (int i = 0; i < numConstraints; i++)
        {
            int varIndex = basicVariables[i];
            double value = rhs[i];
            if (varIndex < originalVars)
            {
                Console.WriteLine($"x{varIndex + 1} = {value:F2}");
            }
            else
            {
                Console.WriteLine($"s{varIndex - originalVars + 1} = {value:F2}");
            }
            optimalValue += objCoeffs[varIndex] * value; // Calculate optimal value
        }

        // Display values for non-basic slack variables (not in basic set)
        for (int i = originalVars; i < totalVars; i++)
        {
            if (!basicSet.Contains(i))
            {
                Console.WriteLine($"s{i - originalVars + 1} = 0.00");
            }
        }

        Console.WriteLine($"\nOptimal Value: {optimalValue:F2}");

        // Calculate and display shadow prices
        Console.WriteLine("\nShadow Prices (Dual Variables):");
        for (int i = 0; i < numConstraints; i++)
        {
            double shadowPrice = objCoeffs[basicVariables[i]];
            Console.WriteLine($"Dual variable for constraint {i + 1}: {shadowPrice:F2}");
        }

        // Calculate and display slack variables correctly
        Console.WriteLine("\nSlack Variables:");
        for (int i = 0; i < numConstraints; i++)
        {
            double slack = rhs[i];
            if (basicVariables[i] < originalVars)
            {
                // The slack is zero for basic variables corresponding to original variables
                slack = 0;
            }
            Console.WriteLine($"Slack for constraint {i + 1}: {slack:F2}");
        }
    }

    // Shared function to display the current tableau
    static void PrintCurrentDictionaryForPrimal(double[] objectiveCoefficients, double[,] lhs, double[] rhs, int[] basicVariables, int totalVars, int numConstraints)
    {
        int originalVars = totalVars - numConstraints;

        Console.WriteLine("\nCurrent Dictionary:");
        Console.WriteLine("Objective Coefficients:");
        for (int j = 0; j < totalVars; j++)
        {
            if (j < originalVars)
                Console.Write($"x{j + 1}: {objectiveCoefficients[j]:F2} ");
            else
                Console.Write($"s{j - originalVars + 1}: {objectiveCoefficients[j]:F2} ");
        }
        Console.WriteLine();

        Console.WriteLine("\nConstraints:");
        for (int i = 0; i < numConstraints; i++)
        {
            Console.Write($"Basic Variable (Row {i + 1}): ");
            if (basicVariables[i] < originalVars)
                Console.Write($"x{basicVariables[i] + 1} = ");
            else
                Console.Write($"s{basicVariables[i] - originalVars + 1} = ");

            Console.Write($"{rhs[i]:F2} ");

            for (int j = 0; j < totalVars; j++)
            {
                if (lhs[i, j] != 0)
                {
                    if (j < originalVars)
                        Console.Write($"+ ({lhs[i, j]:F2}) x{j + 1} ");
                    else
                        Console.Write($"+ ({lhs[i, j]:F2}) s{j - originalVars + 1} ");
                }
            }
            Console.WriteLine();
        }
        Console.WriteLine();
    }


    static void SolveUsingDualSimplex(double[,] lhs, double[] rhs, double[] objectiveCoefficients, int totalVars, int numConstraints)
    {
        Console.WriteLine("\n--- Starting Dual Simplex Method ---");
        bool shouldContinue = true;
        int iterationCount = 0;
        int originalVars = totalVars - numConstraints;

        // Copy the original objective coefficients
        double[] objCoeffs = new double[totalVars];
        Array.Copy(objectiveCoefficients, objCoeffs, totalVars);

        // Initialize basic variables (slack variables should be basic initially)
        int[] basicVariables = new int[numConstraints];
        for (int i = 0; i < numConstraints; i++)
        {
            basicVariables[i] = originalVars + i; // Indices of slack variables
        }

        while (shouldContinue)
        {
            iterationCount++;
            Console.WriteLine($"\n--- Iteration {iterationCount} ---");

            // Step 1: Identify the most negative RHS entry (leaving variable)
            int leavingVarIndex = -1;
            double mostNegativeRHS = 0;
            for (int i = 0; i < numConstraints; i++)
            {
                if (rhs[i] < mostNegativeRHS)
                {
                    mostNegativeRHS = rhs[i];
                    leavingVarIndex = i;
                }
            }

            // If no negative RHS, optimal solution is found
            if (leavingVarIndex == -1)
            {
                Console.WriteLine("Optimal solution found.");
                shouldContinue = false;
                break;
            }

            // Step 2: Select entering variable by dual feasibility ratio test
            int enteringVarIndex = -1;
            double minRatio = double.MaxValue;
            for (int j = 0; j < totalVars; j++)
            {
                if (lhs[leavingVarIndex, j] < 0)  // Only consider negative entries in leaving row
                {
                    double reducedCost = objectiveCoefficients[j];
                    for (int i = 0; i < numConstraints; i++)
                    {
                        reducedCost -= objectiveCoefficients[basicVariables[i]] * lhs[i, j];
                    }
                    double ratio = reducedCost / lhs[leavingVarIndex, j];
                    if (ratio < minRatio)
                    {
                        minRatio = ratio;
                        enteringVarIndex = j;
                    }
                }
            }

            // If no entering variable is found, the problem is infeasible
            if (enteringVarIndex == -1)
            {
                Console.WriteLine("Problem is infeasible.");
                shouldContinue = false;
                break;
            }

            // Print chosen entering and leaving variables with corrected indexing
            Console.WriteLine($"Chosen leaving variable: {(basicVariables[leavingVarIndex] < originalVars ? "x" : "s")}{(basicVariables[leavingVarIndex] < originalVars ? basicVariables[leavingVarIndex] + 1 : basicVariables[leavingVarIndex] - originalVars + 1)}");
            Console.WriteLine($"Chosen entering variable: {(enteringVarIndex < originalVars ? "x" : "s")}{(enteringVarIndex < originalVars ? enteringVarIndex + 1 : enteringVarIndex - originalVars + 1)}");

            // Step 3: Perform pivot operation on the selected row and column
            double pivotElement = lhs[leavingVarIndex, enteringVarIndex];
            if (Math.Abs(pivotElement) < 1e-9)
            {
                Console.WriteLine("Pivot element is too small, leading to numerical instability.");
                shouldContinue = false;
                break;
            }

            // Normalize the pivot row
            for (int j = 0; j < totalVars; j++)
            {
                lhs[leavingVarIndex, j] /= pivotElement;
            }
            rhs[leavingVarIndex] /= pivotElement;

            // Update other rows to zero out the entering column
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

            // Update the objective function coefficients
            double objectiveFactor = objectiveCoefficients[enteringVarIndex];
            for (int j = 0; j < totalVars; j++)
            {
                objectiveCoefficients[j] -= objectiveFactor * lhs[leavingVarIndex, j];
            }

            // Update the basic variables
            basicVariables[leavingVarIndex] = enteringVarIndex;

            // Print the current tableau
            PrintCurrentTableau1(objectiveCoefficients, lhs, rhs, basicVariables, totalVars, numConstraints);
        }

        // Output the final solution
        OutputFinalSolutionWithShadowPricesAndDualSurplus(objCoeffs, lhs, rhs, basicVariables, totalVars, numConstraints);
        
    }

    // Utility method to print the current tableau for dual simplex
    static void PrintCurrentTableau1(double[] objectiveCoefficients, double[,] lhs, double[] rhs, int[] basicVariables, int totalVars, int numConstraints)
    {
        Console.WriteLine("\nCurrent Tableau:");
        Console.WriteLine("Objective Coefficients:");
        for (int j = 0; j < totalVars; j++)
        {
            Console.Write($"{(j < totalVars - numConstraints ? "x" : "s")}{(j < totalVars - numConstraints ? j + 1 : j - (totalVars - numConstraints) + 1)}: {objectiveCoefficients[j]:F2} ");
        }
        Console.WriteLine();

        Console.WriteLine("Constraints:");
        for (int i = 0; i < numConstraints; i++)
        {
            Console.Write($"Basic Variable (Row {i + 1}): {(basicVariables[i] < totalVars - numConstraints ? "x" : "s")}{(basicVariables[i] < totalVars - numConstraints ? basicVariables[i] + 1 : basicVariables[i] - (totalVars - numConstraints) + 1)} = {rhs[i]:F2} ");
            for (int j = 0; j < totalVars; j++)
            {
                Console.Write($"+ ({lhs[i, j]:F2}) {(j < totalVars - numConstraints ? "x" : "s")}{(j < totalVars - numConstraints ? j + 1 : j - (totalVars - numConstraints) + 1)} ");
            }
            Console.WriteLine();
        }
    }

    // Utility method to output the final solution, including shadow prices and dual surplus variables
    static void OutputFinalSolutionWithShadowPricesAndDualSurplus(double[] objCoeffs, double[,] lhs, double[] rhs, int[] basicVariables, int totalVars, int numConstraints)
    {
        Console.WriteLine("\nFinal Solution:");
        double optimalValue = 0;
        int originalVars = totalVars - numConstraints;
        HashSet<int> basicSet = new HashSet<int>(basicVariables);

        // Display values for basic variables and calculate the optimal value
        for (int i = 0; i < numConstraints; i++)
        {
            int varIndex = basicVariables[i];
            double value = rhs[i];
            Console.WriteLine($"{(varIndex < originalVars ? "x" : "s")}{(varIndex < originalVars ? varIndex + 1 : varIndex - originalVars + 1)} = {value:F2}");
            optimalValue += objCoeffs[varIndex] * value;
        }

        Console.WriteLine($"\nOptimal Value: {optimalValue:F2}");

        // Calculate and display shadow prices (dual variables) only for binding constraints
        Console.WriteLine("\nShadow Prices (Dual Variables):");
        for (int i = 0; i < numConstraints; i++)
        {
            // Calculate dual surplus to check if the constraint is binding
            double dualSurplus = rhs[i];
            for (int j = 0; j < totalVars; j++)
            {
                if (!basicSet.Contains(j)) // Only non-basic variables
                {
                    dualSurplus -= lhs[i, j] * objCoeffs[j];
                }
            }

            if (dualSurplus == 0) // Binding constraint
            {
                int basicVarIndex = basicVariables[i];
                double shadowPrice = objCoeffs[basicVarIndex];
                Console.WriteLine($"Shadow price for constraint {i + 1}: {shadowPrice:F2}");
            }
            else // Non-binding constraint
            {
                Console.WriteLine($"Shadow price for constraint {i + 1}: 0.00");
            }
        }

        // Display dual surplus variables for all constraints
        Console.WriteLine("\nDual Surplus Variables:");
        for (int i = 0; i < numConstraints; i++)
        {
            double dualSurplus = rhs[i];
            for (int j = 0; j < totalVars; j++)
            {
                if (!basicSet.Contains(j)) // Only non-basic variables
                {
                    dualSurplus -= lhs[i, j] * objCoeffs[j];
                }
            }
            Console.WriteLine($"Dual surplus for constraint {i + 1}: {dualSurplus:F2}");
        }
    }

    // Solve Using Bland's Rule
    static void SolveUsingBlandsRule(double[,] lhs, double[] rhs,
                                     double[] originalObjCoeffs, int totalVars, int numConstraints)
    {
        bool shouldContinue = true;
        int iterationCount = 0;

        // Copy the original objective coefficients
        double[] objCoeffs = new double[totalVars];
        double[] objectiveCoefficients = new double[totalVars]; // This will be modified during iterations

        Array.Copy(originalObjCoeffs, objCoeffs, totalVars); // Keep a copy of the original coefficients
        Array.Copy(originalObjCoeffs, objectiveCoefficients, totalVars); // This array will be updated

        // Initialize basic variables (slack variables)
        int[] basicVariables = new int[numConstraints];
        for (int i = 0; i < numConstraints; i++)
        {
            basicVariables[i] = totalVars - numConstraints + i; // Indices of slack variables
        }

        while (shouldContinue)
        {
            iterationCount++;
            Console.WriteLine($"Starting iteration {iterationCount} of Bland's Rule.");

            // Perform a single iteration of Bland's Rule
            shouldContinue = ApplyBlandsRule(lhs, rhs, objectiveCoefficients, totalVars, numConstraints, basicVariables, iterationCount);

            if (!shouldContinue)
            {
                Console.WriteLine("Optimal solution reached or no valid pivot available.");
                break;
            }

            // Print the current tableau
            PrintCurrentTableau1(objectiveCoefficients, lhs, rhs, basicVariables, totalVars, numConstraints);
        }

        // Output the final solution
        OutputFinalSolutionWithShadowPricesAndSlacks(objCoeffs, lhs, rhs, basicVariables, totalVars, numConstraints);
      
    }

    static bool ApplyBlandsRule(double[,] lhs, double[] rhs, double[] objectiveCoefficients, int totalVars, int numConstraints, int[] basicVariables, int iterationCount)
    {
        // Print the current tableau (Dictionary)
        PrintCurrentTableau1(objectiveCoefficients, lhs, rhs, basicVariables, totalVars, numConstraints);

        // Calculate the dual variables (pi)
        double[] pi = new double[numConstraints];
        for (int i = 0; i < numConstraints; i++)
        {
            pi[i] = objectiveCoefficients[basicVariables[i]]; // Use the basic variable's objective coefficient
        }

        // Use Bland's Rule to find the entering variable by objective coefficient
        int enteringVarIndex = -1;
        for (int j = 0; j < totalVars; j++)
        {
            double reducedCost = objectiveCoefficients[j];
            for (int i = 0; i < numConstraints; i++)
            {
                reducedCost -= pi[i] * lhs[i, j];
            }
            if (reducedCost > 0) // Check for positive reduced cost
            {
                enteringVarIndex = j; // Choose the smallest index with reducedCost > 0
                break; // Stop as we want the first (smallest index) variable with reducedCost > 0
            }
        }

        // If no entering variable is found, the current solution is optimal
        if (enteringVarIndex == -1)
        {
            Console.WriteLine("No entering variable found. Optimal solution may have been reached.");
            return false; // No more pivots required
        }

        // Print the chosen entering variable
        Console.WriteLine($"Chosen entering variable: x{enteringVarIndex + 1}");

        // Perform the ratio test to find the leaving variable
        int leavingVarIndex = -1;
        double minRatio = double.MaxValue;
        for (int i = 0; i < numConstraints; i++)
        {
            if (lhs[i, enteringVarIndex] > 0) // Positive coefficient for the entering variable
            {
                double ratio = rhs[i] / lhs[i, enteringVarIndex];
                if (ratio < minRatio)
                {
                    minRatio = ratio;
                    leavingVarIndex = i;
                }
                else if (ratio == minRatio) // Apply Bland's Rule to choose the leaving variable
                {
                    if (leavingVarIndex == -1 || basicVariables[i] < basicVariables[leavingVarIndex])
                    {
                        leavingVarIndex = i; // Choose the smallest index for the leaving variable
                    }
                }
            }
        }

        // If no leaving variable is found, the problem is unbounded
        if (leavingVarIndex == -1)
        {
            Console.WriteLine("No leaving variable found. The problem may be unbounded.");
            return false; // No pivot possible, likely an unbounded problem
        }

        // Print the chosen leaving variable
        Console.WriteLine($"Chosen leaving variable: x{basicVariables[leavingVarIndex] + 1}");

        // Perform the pivot operation
        double pivotElement = lhs[leavingVarIndex, enteringVarIndex];

        // Update the leaving row
        for (int j = 0; j < totalVars; j++)
        {
            lhs[leavingVarIndex, j] /= pivotElement;
        }
        rhs[leavingVarIndex] /= pivotElement;

        // Update all other rows
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

        // Update the objective coefficients in-place using the original objective coefficient
        double objectiveFactor = objectiveCoefficients[enteringVarIndex];
        for (int j = 0; j < totalVars; j++)
        {
            objectiveCoefficients[j] -= objectiveFactor * lhs[leavingVarIndex, j];
        }

        // Update the basic variable index
        basicVariables[leavingVarIndex] = enteringVarIndex;

        // Print updated objective coefficients and RHS after pivot
        Console.WriteLine("\nUpdated Objective Function:");
        for (int j = 0; j < totalVars; j++)
        {
            Console.Write($"{objectiveCoefficients[j]:F4} ");
        }
        Console.WriteLine();

        // Print BFS after pivot
        Console.WriteLine("\nUpdated Basic Feasible Solution (BFS) at this iteration:");
        for (int i = 0; i < numConstraints; i++)
        {
            Console.WriteLine($"x{basicVariables[i] + 1} = {rhs[i]:F4}");
        }

        // Output pivot operation
        Console.WriteLine($"Pivot completed: Entering variable x{enteringVarIndex + 1}, Leaving variable is x{basicVariables[leavingVarIndex] + 1}.\n");

        // Print the updated tableau
        PrintCurrentTableau1(objectiveCoefficients, lhs, rhs, basicVariables, totalVars, numConstraints);

        return true; // Successful pivot operation, continue the process
    }


    // Two-Phase Method
    static void TwoPhaseMethod(double[] objCoeffs, double[,] lhs, double[] rhs, int numVars, int numConstraints)
    {
        Console.WriteLine("\n--- Starting Two-Phase Method ---");

        // Record start time
        var watch = Stopwatch.StartNew();

        // Phase 1: Add artificial variables
        int totalVars = numVars + numConstraints; // Original variables + slack variables
        int totalVarsPhase1 = totalVars + numConstraints; // + artificial variables

        double[] cPhase1 = new double[totalVarsPhase1];
        double[,] lhsPhase1 = new double[numConstraints, totalVarsPhase1];
        double[] rhsPhase1 = new double[numConstraints];

        // Objective coefficients for artificial variables in Phase 1 (Minimize sum of artificial variables)
        for (int j = totalVars; j < totalVarsPhase1; j++)
        {
            cPhase1[j] = -1; // Maximization problem
        }

        // Copy original lhs and rhs
        for (int i = 0; i < numConstraints; i++)
        {
            rhsPhase1[i] = rhs[i];
            for (int j = 0; j < totalVars; j++)
            {
                lhsPhase1[i, j] = lhs[i, j];
            }
            // Add artificial variables coefficients
            lhsPhase1[i, totalVars + i] = 1;
        }

        // Initialize basic variables (artificial variables)
        int[] basicVariables = new int[numConstraints];
        for (int i = 0; i < numConstraints; i++)
        {
            basicVariables[i] = totalVars + i; // Indices of artificial variables
        }

        // Start Phase 1
        bool feasible = true;
        while (true)
        {
            // Compute reduced costs
            double[] pi = new double[numConstraints];
            for (int i = 0; i < numConstraints; i++)
            {
                pi[i] = cPhase1[basicVariables[i]];
            }

            double[] reducedCosts = new double[totalVarsPhase1];
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                reducedCosts[j] = cPhase1[j];
                for (int i = 0; i < numConstraints; i++)
                {
                    reducedCosts[j] -= pi[i] * lhsPhase1[i, j];
                }
            }

            // Find entering variable (most positive reduced cost)
            int enteringVarIndex = -1;
            double maxReducedCost = 0;
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                if (reducedCosts[j] > maxReducedCost)
                {
                    maxReducedCost = reducedCosts[j];
                    enteringVarIndex = j;
                }
            }

            // If no entering variable, optimal for Phase 1
            if (enteringVarIndex == -1)
            {
                // Check if the artificial variables are zero
                double objValuePhase1 = 0;
                for (int i = 0; i < numConstraints; i++)
                {
                    objValuePhase1 += cPhase1[basicVariables[i]] * rhsPhase1[i];
                }
                if (Math.Abs(objValuePhase1) > 1e-6)
                {
                    feasible = false;
                }
                break;
            }

            // Perform ratio test to find leaving variable
            int leavingVarIndex = -1;
            double minRatio = double.MaxValue;
            for (int i = 0; i < numConstraints; i++)
            {
                if (lhsPhase1[i, enteringVarIndex] > 1e-6)
                {
                    double ratio = rhsPhase1[i] / lhsPhase1[i, enteringVarIndex];
                    if (ratio < minRatio)
                    {
                        minRatio = ratio;
                        leavingVarIndex = i;
                    }
                }
            }

            // If no leaving variable, problem is unbounded
            if (leavingVarIndex == -1)
            {
                Console.WriteLine("Problem is unbounded in Phase 1.");
                feasible = false;
                break;
            }

            // Pivot operation
            double pivotElement = lhsPhase1[leavingVarIndex, enteringVarIndex];

            // Update the leaving row
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                lhsPhase1[leavingVarIndex, j] /= pivotElement;
            }
            rhsPhase1[leavingVarIndex] /= pivotElement;

            // Update all other rows
            for (int i = 0; i < numConstraints; i++)
            {
                if (i != leavingVarIndex)
                {
                    double factor = lhsPhase1[i, enteringVarIndex];
                    for (int j = 0; j < totalVarsPhase1; j++)
                    {
                        lhsPhase1[i, j] -= factor * lhsPhase1[leavingVarIndex, j];
                    }
                    rhsPhase1[i] -= factor * rhsPhase1[leavingVarIndex];
                }
            }

            // Update the objective coefficients
            double objectiveFactor = cPhase1[enteringVarIndex];
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                cPhase1[j] -= objectiveFactor * lhsPhase1[leavingVarIndex, j];
            }

            // Update basic variables
            basicVariables[leavingVarIndex] = enteringVarIndex;
        }

        if (!feasible)
        {
            Console.WriteLine("The problem is infeasible.");
            return;
        }

        // Remove artificial variables from the basis
        for (int i = 0; i < numConstraints; i++)
        {
            if (basicVariables[i] >= totalVars)
            {
                // Artificial variable is in the basis
                // Try to remove it by finding a non-artificial variable with non-zero coefficient
                int enteringVarIndex = -1;
                for (int j = 0; j < totalVars; j++)
                {
                    if (Math.Abs(lhsPhase1[i, j]) > 1e-6)
                    {
                        enteringVarIndex = j;
                        break;
                    }
                }
                if (enteringVarIndex == -1)
                {
                    Console.WriteLine("Cannot remove artificial variable from basis. Problem is infeasible.");
                    feasible = false;
                    break;
                }
                else
                {
                    // Perform pivot operation to replace artificial variable
                    double pivotElement = lhsPhase1[i, enteringVarIndex];

                    // Normalize pivot row
                    for (int j = 0; j < totalVarsPhase1; j++)
                    {
                        lhsPhase1[i, j] /= pivotElement;
                    }
                    rhsPhase1[i] /= pivotElement;

                    // Update other rows
                    for (int k = 0; k < numConstraints; k++)
                    {
                        if (k != i)
                        {
                            double factor = lhsPhase1[k, enteringVarIndex];
                            for (int j = 0; j < totalVarsPhase1; j++)
                            {
                                lhsPhase1[k, j] -= factor * lhsPhase1[i, j];
                            }
                            rhsPhase1[k] -= factor * rhsPhase1[i];
                        }
                    }

                    // Update the objective function coefficients
                    double objFactor = cPhase1[enteringVarIndex];
                    for (int j = 0; j < totalVarsPhase1; j++)
                    {
                        cPhase1[j] -= objFactor * lhsPhase1[i, j];
                    }

                    // Update basic variable
                    basicVariables[i] = enteringVarIndex;
                }
            }
        }

        if (!feasible)
        {
            Console.WriteLine("The problem is infeasible.");
            return;
        }

        // Prepare for Phase 2 by removing artificial variables
        double[,] lhsPhase2 = new double[numConstraints, totalVars];
        double[] cPhase2 = new double[totalVars];
        double[] rhsPhase2 = new double[numConstraints];

        // Copy the relevant parts from Phase 1 arrays
        for (int i = 0; i < numConstraints; i++)
        {
            rhsPhase2[i] = rhsPhase1[i];
            for (int j = 0; j < totalVars; j++)
            {
                lhsPhase2[i, j] = lhsPhase1[i, j];
            }
        }

        // Set the original objective coefficients
        for (int j = 0; j < totalVars; j++)
        {
            cPhase2[j] = objCoeffs[j];
        }

        // Ensure basic variables are within the correct range
        for (int i = 0; i < numConstraints; i++)
        {
            if (basicVariables[i] >= totalVars)
            {
                Console.WriteLine("Artificial variable still in basis after removal. Problem is infeasible.");
                feasible = false;
                break;
            }
        }

        if (!feasible)
        {
            Console.WriteLine("The problem is infeasible.");
            return;
        }

        // Proceed with Phase 2 using the updated arrays
        lhsPhase1 = lhsPhase2;
        rhsPhase1 = rhsPhase2;
        cPhase1 = cPhase2;
        totalVarsPhase1 = totalVars; // Update totalVarsPhase1 to reflect the removal

        // Start Phase 2
        while (true)
        {
            // Compute reduced costs
            double[] pi = new double[numConstraints];
            for (int i = 0; i < numConstraints; i++)
            {
                pi[i] = cPhase1[basicVariables[i]];
            }

            double[] reducedCosts = new double[totalVarsPhase1];
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                reducedCosts[j] = cPhase1[j];
                for (int i = 0; i < numConstraints; i++)
                {
                    reducedCosts[j] -= pi[i] * lhsPhase1[i, j];
                }
            }

            // Find entering variable (most positive reduced cost)
            int enteringVarIndex = -1;
            double maxReducedCost = 0;
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                if (reducedCosts[j] > maxReducedCost)
                {
                    maxReducedCost = reducedCosts[j];
                    enteringVarIndex = j;
                }
            }

            // If no entering variable, optimal for Phase 2
            if (enteringVarIndex == -1)
            {
                break;
            }

            // Perform ratio test to find leaving variable
            int leavingVarIndex = -1;
            double minRatio = double.MaxValue;
            for (int i = 0; i < numConstraints; i++)
            {
                if (lhsPhase1[i, enteringVarIndex] > 1e-6)
                {
                    double ratio = rhsPhase1[i] / lhsPhase1[i, enteringVarIndex];
                    if (ratio < minRatio)
                    {
                        minRatio = ratio;
                        leavingVarIndex = i;
                    }
                }
            }

            // If no leaving variable, problem is unbounded
            if (leavingVarIndex == -1)
            {
                Console.WriteLine("Problem is unbounded in Phase 2.");
                return;
            }

            // Pivot operation
            double pivotElement = lhsPhase1[leavingVarIndex, enteringVarIndex];

            // Update the leaving row
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                lhsPhase1[leavingVarIndex, j] /= pivotElement;
            }
            rhsPhase1[leavingVarIndex] /= pivotElement;

            // Update all other rows
            for (int i = 0; i < numConstraints; i++)
            {
                if (i != leavingVarIndex)
                {
                    double factor = lhsPhase1[i, enteringVarIndex];
                    for (int j = 0; j < totalVarsPhase1; j++)
                    {
                        lhsPhase1[i, j] -= factor * lhsPhase1[leavingVarIndex, j];
                    }
                    rhsPhase1[i] -= factor * rhsPhase1[leavingVarIndex];
                }
            }

            // Update the objective coefficients
            double objectiveFactor = cPhase1[enteringVarIndex];
            for (int j = 0; j < totalVarsPhase1; j++)
            {
                cPhase1[j] -= objectiveFactor * lhsPhase1[leavingVarIndex, j];
            }

            // Update basic variables
            basicVariables[leavingVarIndex] = enteringVarIndex;
        }

        // Output the final solution
        Console.WriteLine("Optimal Solution:");
        double[] solution = new double[totalVarsPhase1];

        for (int i = 0; i < numConstraints; i++)
        {
            solution[basicVariables[i]] = rhsPhase1[i];
        }

        for (int j = 0; j < totalVars; j++)
        {
            if (Math.Abs(solution[j]) > 1e-6)
            {
                Console.WriteLine($"x{j + 1} = {solution[j]:F4}");
            }
        }

        // Compute the objective value
        double objectiveValue = 0;
        for (int j = 0; j < totalVars; j++)
        {
            objectiveValue += objCoeffs[j] * solution[j];
        }
        Console.WriteLine($"Objective Value: {objectiveValue:F4}");

        // Record end time
        watch.Stop();
        TimeSpan ts = watch.Elapsed;
        Console.WriteLine($"Total runtime for Two-Phase Method: {ts.TotalSeconds} seconds");
    }
}
