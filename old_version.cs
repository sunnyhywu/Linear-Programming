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
            Console.WriteLine("\nvariables for crops types: ");
            int varnum = 0;
            var usedVariables = new HashSet<string>(); // record the variable which has already been show

            foreach (var variable in allVariables)
            {
                Console.Write(variable + " ");
                usedVariables.Add(variable);
                varnum += 1;
                if (varnum == 3)
                {
                    break;
                }
            }

            Console.WriteLine();
            Console.WriteLine("\nvariables for number of selling: ");
            varnum = 0;
            foreach (var variable in allVariables)
            {
                if (!usedVariables.Contains(variable))
                {
                    Console.Write(variable + " ");
                    usedVariables.Add(variable);
                    varnum += 1;
                    if (varnum == 2 * numScenarios)
                    {
                        break;
                    }
                }
            }

            Console.WriteLine();
            Console.WriteLine("\nvariables for number of buying: ");
            varnum = 0;
            foreach (var variable in allVariables)
            {
                if (!usedVariables.Contains(variable))
                {
                    Console.Write(variable + " ");
                }
            }

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

            /*
            //6. L-shaped
            double UB = double.PositiveInfinity; // Upper Bound
            double LB = double.NegativeInfinity; // Lower Bound
            int iteration = 0;

            // L-Shape Method: 迭代過程
            while (Math.Abs(UB - LB) > 1e-6) // 停止條件：UB 和 LB 足夠接近
            {
                iteration++;
                // Check whether is complete recourse
                // 判斷是否為 Complete Recourse
                bool isCompleteRecourse = true;

                // 檢查 Master Variables 是否包含 w 或 y 變數
                foreach (var variable in masterVariables)
                {
                    string trimmedVariable = variable.Trim();
                    if (trimmedVariable.StartsWith("w_") || trimmedVariable.StartsWith("y_"))
                    {
                        isCompleteRecourse = false;
                        break;
                    }
                }

                // 解決 Master Problem
                Console.WriteLine("\nSolving Master Problem...");
                //SolveUsingPrimalSimplex(masterLhs, masterRhs, masterCoefficients, masterTotalVars, masterConstraints);

                // 根據判斷結果進行處理
                if (isCompleteRecourse)
                {
                    Console.WriteLine("\nComplete Recourse detected. Only Optimal Cut will be performed in each iteration.");
                    //optimalcut function

                }
                else
                {
                    Console.WriteLine("\nNot Complete Recourse. Feasibility Cut and Optimal Cut will be performed in each iteration.");
                    //feasibility function
                    //optimalcut function
                }
            }
            */

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
                    allcoefficients[$"y_1_{s}"] = 238.0 / numScenarios;
                    allcoefficients[$"y_2_{s}"] = 210.0 / numScenarios;
                    allcoefficients[$"w_1_{s}"] = -170.0 / numScenarios;
                    allcoefficients[$"w_2_{s}"] = -150.0 / numScenarios;
                    allcoefficients[$"w_3_{s}"] = -36.0 / numScenarios;
                    allcoefficients[$"w_4_{s}"] = -10.0 / numScenarios;

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
                    variables.Add($"y_1_{s}");
                    variables.Add($"y_2_{s}");
                }

                for (int s = 1; s <= numScenarios; s++)
                {
                    variables.Add($"w_1_{s}");
                    variables.Add($"w_2_{s}");
                    variables.Add($"w_3_{s}");
                    variables.Add($"w_4_{s}");
                }
                return variables;
            }

            static double[,] GenerateLhsMatrix(List<string> variables, int numScenarios)
            {
                int numConstraints = 1 + numScenarios * 4; // 1 total land constraint + 3 constraints per scenario + 1 notdifferent constraint
                double[,] lhsMatrix = new double[numConstraints, variables.Count];

                // Total land constraint: x1 + x2 + x3 <= 500
                lhsMatrix[0, variables.IndexOf("x_1")] = 1;
                lhsMatrix[0, variables.IndexOf("x_2")] = 1;
                lhsMatrix[0, variables.IndexOf("x_3")] = 1;

                // Generate coefficients for each scenario
                for (int s = 1; s <= numScenarios; s++)
                {
                    int row = 1 + (s - 1) * 4;

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
                    // Coefficient of w3
                    lhsMatrix[row + 3, variables.IndexOf($"w_3_{s}")] = 1; 

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
