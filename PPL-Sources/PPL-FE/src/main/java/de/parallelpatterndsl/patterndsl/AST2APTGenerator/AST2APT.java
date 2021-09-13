package de.parallelpatterndsl.patterndsl.AST2APTGenerator;

import de.monticore.expressions.commonexpressions._ast.*;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.monticore.literals.literals._ast.ASTIntLiteral;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DynamicProgrammingDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.MapDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.StencilDataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.AdditionalArguments;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaList;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.AdditionalArguments.MetaValue;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;


/**
 * Class that implements the generation of the abstract pattern tree based on the abstract syntax tree.
 */
public class AST2APT{

    private ArrayList<IRLExpression> simpleExpressions = new ArrayList<>();

    private HashMap<String, FunctionNode> functionTable = new HashMap<>();

    private HashMap<String, ASTFunction> astFunctionTable = new HashMap<>();

    private HashMap<String, Data> variableData = new HashMap<>();

    private GlobalScope symbolTable;

    private ASTModule module;

    private ArrayList<ParallelCallNode> parallelCallNodes = new ArrayList<>();

    private APTExpressionPrinter printer = new APTExpressionPrinter();

    private ArrayList<IRLExpression> globalAssignments = new ArrayList<>();


    public AST2APT(GlobalScope symbolTable, ASTModule module, int randomIndexLength) {
        this.symbolTable = symbolTable;
        this.module = module;
        RandomStringGenerator.setN(randomIndexLength);
    }

    public AbstractPatternTree generate() {
        //instantiate symbol tables
        FillASTFunctionTable();
        variableData = generateGlobalVariableTable(module);
        functionTable = generateFunctionTable(module);
        AbstractPatternTree.setFunctionTable(functionTable);
        globalAssignments = generateGlobalVariableAssignments(module);


        //get the main node
        MainNode mainNode = (MainNode) functionTable.get("main");

        System.out.println("symbol table finished!");

        //generate function nodes and top down tree structure
        for (ASTFunction astFunction : astFunctionTable.values()) {
            generateFunctionNode(astFunction);
        }

        System.out.println("tree structure finished!");

        //generate the bottom up structure
        for (FunctionNode function : functionTable.values()) {
            generateParentAwareness(function);
        }

        System.out.println("bottom up structure finished!");

        //generate data traces
        APTDataTraceGenerator traceGenerator = new APTDataTraceGenerator();
        traceGenerator.generateTraces(mainNode);

        System.out.println("data trace generation finished!");

        //generate input and output data (accesses) for all nodes
        for (FunctionNode node: functionTable.values() ) {
            generateParentAwareDataAccesses(node);
        }

        System.out.println("parent aware data trace generation finished!");

        //generates the value for functions, if they have parallel ancestors.
        generateHasParallelDescendants();
        System.out.println("parallel descendant computation finished!");

        //generate additional Meta informations.
        generateAdditionalArguments();
        System.out.println("additional argument generation finished!");

        return new AbstractPatternTree(mainNode, variableData, globalAssignments);
    }

    /************************************************
     *
     *
     * Pre-processing Functions
     *
     *
     ************************************************/
    private void FillASTFunctionTable(){
        for (ASTDefinition definition: module.getDefinitionList()) {
            if (definition instanceof ASTFunction) {
                ASTFunction function = (ASTFunction) definition;
                astFunctionTable.put(function.getName(),function);
            }
        }
    }

    /**
     * Function that generates the Global variable table (symbol table).
     * @param ast
     * @return
     */
    private HashMap<String, Data> generateGlobalVariableTable(ASTModule ast) {
        HashMap<String, Data> result = new HashMap<>();
        for (ASTDefinition definition: ast.getDefinitionList()) {
            if (definition instanceof ASTVariable) {
                ASTVariable variable = (ASTVariable) definition;
                Optional<VariableSymbol> variableSymbolOPT = variable.getEnclosingScope().resolve(variable.getName(),VariableSymbol.KIND);
                Data dataValue;
                if (variableSymbolOPT.isPresent()) {
                    VariableSymbol variableSymbol = variableSymbolOPT.get();
                    if (variableSymbol.getType() instanceof ASTListType) {
                        dataValue = new ArrayData(variableSymbol.getName(), APTTypesPrinter.printType(variableSymbol.getType()), false, variableSymbol.getShape(), variableSymbol.isArrayOnStack());
                        result.put(variableSymbol.getName(),dataValue);
                    } else {
                        dataValue = new PrimitiveData(variableSymbol.getName(),APTTypesPrinter.printType(variableSymbol.getType()), false);
                        result.put(variableSymbol.getName(),dataValue);
                    }
                }
            }
        }
        return result;
    }

    /**
     * Generates the static assignment expressions for the initialization of the global variables.
     * @param ast
     * @return
     */
    private ArrayList<IRLExpression> generateGlobalVariableAssignments(ASTModule ast) {
        ArrayList<IRLExpression> result = new ArrayList<>();
        for (ASTDefinition definition: ast.getDefinitionList()) {
            if (definition instanceof ASTVariable) {
                ASTVariable variable = (ASTVariable) definition;
                if (variable.isPresentExpression()) {
                    if (variableData.containsKey(variable.getName())) {
                        Data data = variableData.get(variable.getName());
                        AssignmentExpression globalAssignment = new AssignmentExpression(data, new ArrayList<>(), (OperationExpression) printer.printExpression(((ASTVariable) definition).getExpression(), variableData,astFunctionTable), Operator.ASSIGNMENT);
                        result.add(globalAssignment);
                    }
                }
            }
        }
        return result;
    }

    /**
     * Function that creates the Function table (symbol table).
     * @param ast
     * @return
     */
    private HashMap<String, FunctionNode> generateFunctionTable(ASTModule ast) {
        HashMap<String, FunctionNode> result = new HashMap<>();
        for (ASTDefinition definition: ast.getDefinitionList()) {
            if (definition instanceof ASTFunction) {
                ASTFunction astFunction = (ASTFunction) definition;
                if (astFunction.getPatternType().isPresentStencil()) {
                    int dimension = getDimensionality((ASTListType) astFunction.getFunctionParameter().getType(),1);
                    StencilNode stencilNode = new StencilNode(astFunction.getName(), dimension);
                    result.put(astFunction.getName(),stencilNode);
                } else if (astFunction.getPatternType().isPresentMap()) {
                    MapNode mapNode = new MapNode(astFunction.getName());
                    result.put(astFunction.getName(),mapNode);
                } else if (astFunction.getPatternType().isPresentReduction()) {
                    ReduceNode reduceNode = new ReduceNode(astFunction.getName());
                    result.put(astFunction.getName(), reduceNode);
                } else if (astFunction.getPatternType().isPresentSerial() && astFunction.getName().equals("main")) {
                    MainNode mainNode = new MainNode(astFunction.getName(),PrimitiveDataTypes.INTEGER_32BIT,false);
                    result.put(astFunction.getName(),mainNode);
                } else if (astFunction.getPatternType().isPresentSerial()) {
                    boolean islist = false;
                    if (astFunction.getType() instanceof ASTListType) {
                        islist = true;
                    }
                    SerialNode serialNode = new SerialNode(astFunction.getName(), APTTypesPrinter.printType(astFunction.getType()), islist);
                    result.put(astFunction.getName(),serialNode);
                } else if (astFunction.getPatternType().isPresentRecursion()) {
                    RecursionNode recursionNode = new RecursionNode(astFunction.getName());
                    result.put(astFunction.getName(),recursionNode);
                } else if (astFunction.getPatternType().isPresentDynamicProgramming()) {
                    DynamicProgrammingNode dynamicProgrammingNode = new DynamicProgrammingNode(astFunction.getName(), getDimensionality((ASTListType) astFunction.getFunctionParameter().getType(),1) + 1);
                    result.put(astFunction.getName(),dynamicProgrammingNode);
                } else {
                    Log.error("Pattern type not recognized" + astFunction.get_SourcePositionStart());
                    throw new RuntimeException("Critical error!");
                }
            }
        }
        return result;
    }

    /************************************************
     *
     *
     * Top-down tree generation Functions
     *
     *
     ************************************************/

    /**
     * Generates the tree structure for the already available function node given its AST counterpart.
     * @param astFunction
     */
    private void generateFunctionNode(ASTFunction astFunction) {
        HashMap<String, Data> oldVariableData = new HashMap<>(variableData);
        ArrayList<PatternNode> childNodes = new ArrayList<>();
        ArrayList<Data> parameters = new ArrayList<>();
        FunctionNode functionNode = functionTable.get(astFunction.getName());

        functionNode.setArgumentCount(astFunction.getFunctionParameters().sizeFunctionParameters());

        // Initialize parameter as data elements
        for (ASTFunctionParameter functionParameter: astFunction.getFunctionParameters().getFunctionParameterList()) {
            if (functionParameter.getType() instanceof ASTTypeName) {
                PrimitiveData parameter = new PrimitiveData(functionParameter.getName(),APTTypesPrinter.printType(functionParameter.getType()),true);
                parameters.add(parameter);
                variableData.put(parameter.getIdentifier(),parameter);
                functionNode.addArgumentValues(parameter);
            } else if (functionParameter.getType() instanceof ASTListType) {
                ASTType type  = functionParameter.getType();
                ArrayList<Integer> shapeDummy = new ArrayList<>();
                while(type instanceof ASTListType) {
                    shapeDummy.add(-1);
                    type = ((ASTListType) type).getType();
                }
                ArrayData parameter = new ArrayData(functionParameter.getName(),APTTypesPrinter.printType(functionParameter.getType()),true, shapeDummy, false);
                parameters.add(parameter);
                variableData.put(functionParameter.getName(),parameter);
                functionNode.addArgumentValues(parameter);
            }
        }

        // Instantiate the return parameter for parallel functions
        if (!astFunction.getPatternType().isPresentSerial()) {

            FunctionNode node = functionTable.get(astFunction.getName());
            //Add INDEX to the function table
            if (node instanceof StencilNode) {
                for (int i = 0; i < ((StencilNode) node).getDimension(); i++) {
                    variableData.put("INDEX" + i, new PrimitiveData("INDEX" + i, PrimitiveDataTypes.INTEGER_32BIT, true));
                }
            } else if (node instanceof DynamicProgrammingNode) {
                for (int i = 0; i < ((DynamicProgrammingNode) node).getDimension(); i++) {
                    variableData.put("INDEX" + i, new PrimitiveData("INDEX" + i, PrimitiveDataTypes.INTEGER_32BIT, true));
                }
            } else {
                variableData.put("INDEX", new PrimitiveData("INDEX", PrimitiveDataTypes.INTEGER_32BIT, true));
            }


            if (node instanceof ParallelNode) {
                ParallelNode parallelNode = (ParallelNode) node;
                ASTFunctionParameter returnParameter = astFunction.getFunctionParameter();
                if (returnParameter.getType() instanceof ASTTypeName) {
                    PrimitiveData parameter = new PrimitiveData(returnParameter.getName(),APTTypesPrinter.printType(returnParameter.getType()), true,true);
                    parallelNode.setReturnElement(parameter);
                    variableData.put(parameter.getIdentifier(),parameter);
                } else if (returnParameter.getType() instanceof ASTListType) {
                    ASTType type  = returnParameter.getType();
                    ArrayList<Integer> shapeDummy = new ArrayList<>();
                    while(type instanceof ASTListType) {
                        shapeDummy.add(-1);
                        type = ((ASTListType) type).getType();
                    }
                    ArrayData parameter = new ArrayData(returnParameter.getName(),APTTypesPrinter.printType(returnParameter.getType()), true,true, shapeDummy, false);
                    parallelNode.setReturnElement(parameter);
                    variableData.put(parameter.getIdentifier(),parameter);
                }
            }
        }


        childNodes = generateBlockStatement(astFunction.getBlockStatement());

        functionNode.setChildren(childNodes);
        functionNode.setVariableTable(variableData);
        variableData = oldVariableData;
    }

    private ReturnNode generateReturnNode(ASTReturnStatement returnStatement) {
        ReturnNode returnNode = new ReturnNode();
        ArrayList<PatternNode> childNodes = new ArrayList<>();

        if (returnStatement.isPresentReturnExpression()) {
            childNodes.add(generateComplexExpressionNode(printer.printExpression(returnStatement.getReturnExpression(),variableData,astFunctionTable),getFunctionCalls(returnStatement.getReturnExpression())));
        }

        returnNode.setChildren(childNodes);
        returnNode.setVariableTable(variableData);

        return returnNode;
    }

    /**
     * Generates the list of child nodes from a given block-statement.
     * @param blockStatement
     * @return
     */
    private ArrayList<PatternNode> generateBlockStatement(ASTBlockStatement blockStatement) {
        ArrayList<PatternNode> result = new ArrayList<>();
        for (ASTBlockElement element: blockStatement.getBlockElementList() ) {
            //generate singular expressions
            if (element.isPresentExpression()) {
                ASTExpression expression = element.getExpression();
                //test whether a simple expression block node or a complex expression is applicable
                if (getFunctionCalls(expression).size() != 0) {
                    if (simpleExpressions.size() != 0) {
                        result.add(generateSimpleExpressionBlockNode());
                    }
                    result.add(generateComplexExpressionNode(printer.printExpression(expression,variableData,astFunctionTable), getFunctionCalls(expression)));
                } else {
                    simpleExpressions.add(printer.printExpression(expression,variableData,astFunctionTable));
                }
            } else
                //generate variables
                if (element.isPresentVariable()) {
                ASTVariable variable = element.getVariable();
                Data dataElement;
                //test whether the variable is an array or a primitive value
                if (variable.getType() instanceof ASTListType) {
                    ArrayList<Integer> shape = new ArrayList<>();
                    boolean onStack = false;
                    Optional<VariableSymbol> variableSymbolOPT = variable.getEnclosingScope().resolve(variable.getName(),VariableSymbol.KIND);
                    if (variableSymbolOPT.isPresent()) {
                        VariableSymbol symbol = variableSymbolOPT.get();
                        shape = symbol.getShape();
                        onStack = symbol.isArrayOnStack();
                    }
                    dataElement = new ArrayData(variable.getName(),APTTypesPrinter.printType(variable.getType()), false, shape, onStack);
                } else {
                    dataElement = new PrimitiveData(variable.getName(),APTTypesPrinter.printType(variable.getType()), false);
                }

                variableData.put(variable.getName(),dataElement);

                //generate the initialization expression
                if (variable.isPresentExpression()) {
                    ASTExpression expression = variable.getExpression();
                    AssignmentExpression assignmentExpression = new AssignmentExpression(dataElement, new ArrayList<>(), (OperationExpression) printer.printExpression(expression,variableData,astFunctionTable), Operator.ASSIGNMENT);
                    if (getFunctionCalls(expression).size() != 0) {
                        if (simpleExpressions.size() != 0) {
                            result.add(generateSimpleExpressionBlockNode());
                        }

                        result.add(generateComplexExpressionNode(assignmentExpression,getFunctionCalls(expression)));
                    } else {
                        simpleExpressions.add(assignmentExpression);
                    }
                }
            } else
                //generate control statement nodes
                if (element.isPresentStatement()) {
                    if (simpleExpressions.size() != 0) {
                        result.add(generateSimpleExpressionBlockNode());
                    }
                    ASTStatement statement = element.getStatement();
                    if (statement instanceof ASTIfStatement) {
                        result.add(generateBranchNode((ASTIfStatement) statement));
                    } else if (statement instanceof ASTWhileStatement) {
                        result.add(generateWhileLoopNode((ASTWhileStatement) statement));
                    } else if (statement instanceof ASTForStatement) {
                        result.add(generateForLoopNode((ASTForStatement) statement));
                    } else if (statement instanceof ASTPatternCallStatement) {
                        result.add(generateParallelCallNode((ASTPatternCallStatement) statement));
                    } else if (statement instanceof ASTReturnStatement) {
                        result.add(generateReturnNode((ASTReturnStatement) statement));
                    }
                }
        }
        if (simpleExpressions.size() != 0) {
            result.add(generateSimpleExpressionBlockNode());
        }
        return result;
    }

    /**
     * Generates the branching behavior in the APT from a given AST Node.
     * @param astIfStatement
     * @return
     */
    private BranchNode generateBranchNode(ASTIfStatement astIfStatement) {
        BranchNode branchNode = new BranchNode();
        ArrayList<PatternNode> childNodes = new ArrayList<>();

        ASTStatement astStatement = astIfStatement;
        while (astStatement instanceof ASTIfStatement) {
            ASTIfStatement caseNode = (ASTIfStatement) astStatement;
            childNodes.add(generateBranchCaseNode(caseNode.getThenStatement(),Optional.of(caseNode.getCondition())));
            if (caseNode.isPresentElseStatement()) {
                // Test whether the else Branch is pure or another if branch
                if (caseNode.getElseStatement().isPresentIfStatement()) {
                    astStatement = caseNode.getElseStatement().getIfStatement();
                } else {
                    childNodes.add(generateBranchCaseNode(caseNode.getElseStatement().getBlockStatement(),Optional.empty()));
                    break;
                }
            }
            // Break if no pure else branch is given
            else {
                break;
            }
        }

        branchNode.setVariableTable(variableData);
        branchNode.setChildren(childNodes);
        return branchNode;
    }

    /**
     * Generates a single branch with its given condition (if available).
     * @param astBlockStatement
     * @param astExpressionOPT
     * @return
     */
    private BranchCaseNode generateBranchCaseNode(ASTBlockStatement astBlockStatement, Optional<ASTExpression> astExpressionOPT) {
        HashMap<String, Data> oldVariableData = new HashMap<>(variableData);
        ArrayList<PatternNode> childNodes = new ArrayList<>();
        boolean hasCondition = false;


        if (astExpressionOPT.isPresent()) {
            ASTExpression astExpression = astExpressionOPT.get();
            childNodes.add(generateComplexExpressionNode(printer.printExpression(astExpression,variableData,astFunctionTable),getFunctionCalls(astExpression)));
            hasCondition = true;
        }

        BranchCaseNode result = new BranchCaseNode(hasCondition);

        childNodes.addAll(generateBlockStatement(astBlockStatement));

        result.setChildren(childNodes);
        result.setVariableTable(variableData);
        variableData = oldVariableData;
        return result;
    }

    /**
     * Generates a complex expression node.
     * @param Expression
     * @param calls
     * @return
     */
    private ComplexExpressionNode generateComplexExpressionNode(IRLExpression Expression, ArrayList<ASTCallExpression> calls) {
        ArrayList<PatternNode> childNodes = new ArrayList<>();
        ComplexExpressionNode complexExpressionNode = new ComplexExpressionNode(Expression);

        for (ASTCallExpression callExpression: calls) {
            childNodes.add(generateCallNode(callExpression));
        }

        complexExpressionNode.setVariableTable(variableData);
        complexExpressionNode.setChildren(childNodes);

        if (complexExpressionNode.getChildren().size() > 0) {
            connectCall2Expression(complexExpressionNode.getExpression(), complexExpressionNode.getChildren(), 0);
        }

        return complexExpressionNode;
    }

    /**
     * Generates a for loop node based on the given statement.
     * @param astForStatement
     * @return
     */
    private LoopNode generateForLoopNode(ASTForStatement astForStatement) {
        HashMap<String, Data> oldVariableData = new HashMap<>(variableData);
        ArrayList<PatternNode> childNodes = new ArrayList<>();

        LoopNode result;

        //generate standard for-loop
        if (astForStatement.getForControl() instanceof ASTCommonForControl) {
            ASTCommonForControl control = (ASTCommonForControl) astForStatement.getForControl();
            AssignmentExpression controlVariableInit;

            if (control.getForInit().isPresentVariable()) {
                PrimitiveData controlVariable = new PrimitiveData(control.getForInit().getVariable().getName(), APTTypesPrinter.printType(control.getForInit().getVariable().getType()), true);
                variableData.put(control.getForInit().getVariable().getName(), controlVariable);

                result = new ForLoopNode(controlVariable);

                OperationExpression initValue = (OperationExpression) printer.printExpression(control.getForInit().getVariable().getExpression(),variableData,astFunctionTable);
                controlVariableInit = new AssignmentExpression(controlVariable, new ArrayList<>(), initValue, Operator.ASSIGNMENT);
                childNodes.add(generateComplexExpressionNode(controlVariableInit, getFunctionCalls(control.getForInit().getVariable().getExpression())));
            } else {
                controlVariableInit = (AssignmentExpression) printer.printExpression(control.getForInit().getExpression(),variableData,astFunctionTable);
                result = new ForLoopNode(controlVariableInit.getOutputElement());
                childNodes.add(generateComplexExpressionNode(controlVariableInit, getFunctionCalls(control.getForInit().getExpression())));
            }

            childNodes.add(generateComplexExpressionNode(printer.printExpression(control.getCondition(),variableData,astFunctionTable), getFunctionCalls(control.getCondition())));
            childNodes.add(generateComplexExpressionNode(printer.printExpression(control.getExpression(),variableData,astFunctionTable), getFunctionCalls(control.getExpression())));
        } else {
            //generate for-each-loop
            Data loopControlVariable;
            if (((ASTForEachControl) astForStatement.getForControl()).getVariable().getType() instanceof ASTListType) {
                int dimension = getDimensionality((ASTListType) ((ASTForEachControl) astForStatement.getForControl()).getVariable().getType(), 0);
                ArrayList<Integer> shape = new ArrayList<>();
                for (int i = 0; i < dimension; i++) {
                    shape.add(-1);
                }
                loopControlVariable = new ArrayData(((ASTForEachControl) astForStatement.getForControl()).getVariable().getName(), APTTypesPrinter.printType(((ASTForEachControl) astForStatement.getForControl()).getVariable().getType()), true, shape, false);
            } else {
                loopControlVariable = new PrimitiveData(((ASTForEachControl) astForStatement.getForControl()).getVariable().getName(), APTTypesPrinter.printType(((ASTForEachControl) astForStatement.getForControl()).getVariable().getType()), true);
            }

            variableData.put(loopControlVariable.getIdentifier(), loopControlVariable);

            IRLExpression expression = printer.printExpression(((ASTForEachControl) astForStatement.getForControl()).getExpression(),variableData,astFunctionTable);
            childNodes.add(generateComplexExpressionNode(expression,getFunctionCalls(((ASTForEachControl) astForStatement.getForControl()).getExpression())));

            result = new ForEachLoopNode(loopControlVariable);
        }

        childNodes.addAll(generateBlockStatement(astForStatement.getBlockStatement()));

        result.setChildren(childNodes);
        result.setVariableTable(variableData);
        variableData = oldVariableData;
        return result;
    }

    /**
     * Generates a while loop node from a given while statement.
     * @param astWhileStatement
     * @return
     */
    private LoopNode generateWhileLoopNode(ASTWhileStatement astWhileStatement) {
        HashMap<String, Data> oldVariableData = new HashMap<>(variableData);
        ArrayList<PatternNode> childNodes = new ArrayList<>();

        WhileLoopNode result = new WhileLoopNode();

        childNodes.add(generateComplexExpressionNode(printer.doPrintExpression(astWhileStatement.getCondition()), getFunctionCalls(astWhileStatement.getCondition())));

        childNodes.addAll(generateBlockStatement(astWhileStatement.getBlockStatement()));

        result.setChildren(childNodes);
        result.setVariableTable(variableData);
        variableData = oldVariableData;
        return result;
    }

    /**
     * Generates a call node for a given function call.
     * @param astCallExpression
     * @return
     */
    private CallNode generateCallNode(ASTCallExpression astCallExpression) {
        if (astCallExpression.getExpression() instanceof ASTNameExpression) {
            ASTNameExpression nameExpression = ((ASTNameExpression) astCallExpression.getExpression());
            CallNode callNode = new CallNode(astCallExpression.getArguments().sizeExpressions(), nameExpression.getName());
            callNode.setVariableTable(variableData);
            return callNode;
        }
        Log.error("Function name not recognized " + astCallExpression.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    /**
     * Generates a parallel call node based on the given statement.
     * @param astPatternCallStatement
     * @return
     */
    private ParallelCallNode generateParallelCallNode(ASTPatternCallStatement astPatternCallStatement) {
        ArrayList<PatternNode> childNodes = new ArrayList<>();

        // Generate additional arguments
        ArrayList<AdditionalArguments> additionalArguments = new ArrayList<>();
        for (ASTExpression expression: astPatternCallStatement.getArgsList()) {
            if( expression instanceof ASTLitExpression) {
                // Handle single literals

                if (((ASTLitExpression) expression).getLiteral() instanceof ASTIntLiteral) {
                    ASTIntLiteral literal = (ASTIntLiteral) ((ASTLitExpression) expression).getLiteral();
                    MetaValue<Long> argument = new MetaValue<>((long) literal.getValue());
                    additionalArguments.add(argument);

                } else {
                    Log.error("Additional arguments must be (lists of) literals of type Int! " + astPatternCallStatement.get_SourcePositionStart());
                    throw new RuntimeException("Critical error!");
                }

            } else if (expression instanceof ASTListExpression) {
                //Handle lists of literals
                ArrayList<Integer> list = new ArrayList<>();
                for (ASTExpression element: ((ASTListExpression) expression).getExpressionList()){
                    if( element instanceof ASTLitExpression) {
                        // Handle single literals

                        if (((ASTLitExpression) element).getLiteral() instanceof ASTIntLiteral) {
                            ASTIntLiteral literal = (ASTIntLiteral) ((ASTLitExpression) element).getLiteral();
                            list.add(literal.getValue());

                        } else {
                            Log.error("Additional arguments must be (lists of) literals of type Int! " + astPatternCallStatement.get_SourcePositionStart());
                            throw new RuntimeException("Critical error!");
                        }

                    } else {
                        Log.error("Additional arguments must be (lists of) literals of type Int! " + astPatternCallStatement.get_SourcePositionStart());
                        throw new RuntimeException("Critical error!");
                    }
                }

                MetaList<Integer> argument = new MetaList<>(list);
                additionalArguments.add(argument);
            } else {
                Log.error("Additional arguments must be (lists of) literals of type Int! " + astPatternCallStatement.get_SourcePositionStart());
                throw new RuntimeException("Critical error!");
            }
        }

        /*
        for (ASTExpression expression: astPatternCallStatement.getArgsList()) {
            childNodes.add(generateComplexExpressionNode((OperationExpression) printer.printExpression(expression,variableData,astFunctionTable), getFunctionCalls(expression)));
        }
         */

        // Generate assignment
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();

        ASTExpression left = astPatternCallStatement.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data = new LiteralData<>("1",PrimitiveDataTypes.COMPLEX_TYPE,1);
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = printer.getAccessScheme((ASTIndexAccessExpression) left, variableData, astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
        }

        String name = astPatternCallStatement.getName();
        PrimitiveDataTypes type = APTTypesPrinter.printType(astFunctionTable.get(name).getFunctionParameter().getType());
        FunctionReturnData returnData = new FunctionReturnData(name,type);
        int dimension = printer.getDimensionality(astFunctionTable.get(name).getFunctionParameter().getType(), 0);

        operandList.add(returnData);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        //generate arguments
        for (int i = 0; i < astPatternCallStatement.getArguments().getExpressionList().size(); i++) {
            ASTExpression expression = astPatternCallStatement.getArguments().getExpressionList().get(i);
            OperationExpression argumentExpression = (OperationExpression) printer.printExpression(expression,variableData,astFunctionTable);
            operandList.addAll(argumentExpression.getOperands());
            operatorList.addAll(argumentExpression.getOperators());
            if (i != astPatternCallStatement.getArguments().getExpressionList().size() - 1){
                operatorList.add(Operator.COMMA);
            }
        }
        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        OperationExpression operation = new OperationExpression(operandList,operatorList);

        AssignmentExpression assignment = new AssignmentExpression(data,accessScheme,operation,Operator.ASSIGNMENT);

        childNodes.add(generateComplexExpressionNode(assignment,getFunctionCalls(astPatternCallStatement)));

        ParallelCallNode result = new ParallelCallNode(astPatternCallStatement.getArguments().sizeExpressions(), astPatternCallStatement.getName(), 0);

        result.setVariableTable(variableData);
        result.setChildren(childNodes);
        result.setAdditionalArguments(additionalArguments);

        parallelCallNodes.add(result);
        return result;
    }

    /**
     * Generates a simple expression block node based on the elements within the simple Expression list.
     * @return
     */
    private SimpleExpressionBlockNode generateSimpleExpressionBlockNode() {
        SimpleExpressionBlockNode simpleExpressionBlockNode = new SimpleExpressionBlockNode(new ArrayList<>(simpleExpressions));
        simpleExpressions.clear();
        simpleExpressionBlockNode.setChildren(new ArrayList<>());
        simpleExpressionBlockNode.setVariableTable(variableData);
        return simpleExpressionBlockNode;
    }



    /************************************************
     *
     *
     * Bottom up structure generation Functions
     *
     *
     ************************************************/

    /**
     * Sets the parent node for all descendants of the given node.
     * @param node
     */
    private void generateParentAwareness(PatternNode node) {
        if (node instanceof CallNode) {
            if (node instanceof ParallelCallNode) {
                for (int i = 0; i <= ((ParallelCallNode) node).getAdditionalArgumentCount(); i++) {
                    node.getChildren().get(i).setParent(node);
                    generateParentAwareness(node.getChildren().get(i));
                }
            }
            return;
        }

        for (PatternNode child : node.getChildren()) {
            child.setParent(node);
            generateParentAwareness(child);
        }
    }


    /************************************************
     *
     *
     * Post-processing Functions (hasParallelAncestors)
     *
     *
     ************************************************/

    private void generateHasParallelDescendants() {

        hasParallelDescendants visitor = new hasParallelDescendants();
        for (FunctionNode node: functionTable.values() ) {
            if (visitor.getResult(node)) {
                node.setHasParallelDescendants(true);
            } else {
                node.setHasParallelDescendants(false);
            }
        }
    }

    private class hasParallelDescendants implements APTVisitor {
        private boolean result = false;

        public boolean getResult(FunctionNode node) {
            result = false;

            node.accept(getRealThis());

            return result;
        }

        @Override
        public void visit(ParallelCallNode node) {
            result = true;
        }

        /**
         * Visitor support functions.
         */
        private APTVisitor realThis = this;

        @Override
        public APTVisitor getRealThis() {
            return realThis;
        }

        @Override
        public void setRealThis(APTVisitor realThis) {
            this.realThis = realThis;
        }


    }


    /************************************************
     *
     *
     * Data access generation Functions
     *
     *
     ************************************************/


    private void generateParentAwareDataAccesses(PatternNode node) {
        if (node instanceof SimpleExpressionBlockNode || node instanceof ComplexExpressionNode) {
            return;
        } else {
            ArrayList<Data> inputData = node.getInputElements();
            ArrayList<Data> outputData = node.getOutputElements();
            ArrayList<DataAccess> inputAccesses = node.getInputAccesses();
            ArrayList<DataAccess> outputAccesses = node.getOutputAccesses();

            int numChildren = node.getChildren().size();
            if (node instanceof ParallelCallNode) {
                numChildren = ((ParallelCallNode) node).getAdditionalArgumentCount() + 1;
            }

            for (int i = 0; i < numChildren; i++) {

                PatternNode childnode = node.getChildren().get(i);
                generateParentAwareDataAccesses(childnode);

                // Update input data
                for (Data element : childnode.getInputElements() ) {
                    if (node.getVariableTable().containsValue(element) && !inputData.contains(element)) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element)) {
                                inputData.add(element);
                            }
                        } else {
                            inputData.add(element);
                        }
                    }
                }
                node.setInputElements(inputData);

                // Update output data
                for (Data element : childnode.getOutputElements() ) {
                    if (node.getVariableTable().containsValue(element) && !outputData.contains(element)) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element)) {
                                outputData.add(element);
                            }
                        } else {
                            outputData.add(element);
                        }
                    }
                }
                node.setOutputElements(outputData);

                // Update input data accesses
                for (DataAccess element : childnode.getInputAccesses() ) {
                    if (node.getVariableTable().containsValue(element.getData())) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element)) {
                                inputAccesses.add(element);
                            }
                        } else {
                            inputAccesses.add(element);
                        }
                    }
                }
                node.setInputAccesses(inputAccesses);

                // Update output data accesses
                for (DataAccess element : childnode.getOutputAccesses() ) {
                    if (node.getVariableTable().containsValue(element.getData())) {
                        if (!(node instanceof FunctionNode)) {
                            if (node.getParent().getVariableTable().containsValue(element)) {
                                outputAccesses.add(element);
                            }
                        } else {
                            outputAccesses.add(element);
                        }
                    }
                }
                node.setOutputAccesses(outputAccesses);
            }
        }
    }


    /************************************************
     *
     *
     * Generate not provided additional arguments.
     *
     *
     ************************************************/

    /**
     * Support class used to generate the computable additional arguments for all parallel calls.
     */
    private class AdditionalArgumentGenerator implements ExtendedShapeAPTVisitor {

        public AdditionalArgumentGenerator() {
        }

        @Override
        public void visit(ParallelCallNode node) {
            String name = node.getFunctionIdentifier();

            FunctionNode function = AbstractPatternTree.getFunctionTable().get(name);

            ArrayList<AdditionalArguments> additionalArguments = new ArrayList<>();

            if (function instanceof MapNode) {
                MapNode mapNode = (MapNode) function;

                MetaValue<Long> start = new MetaValue<>(getMapStart(mapNode,node));

                MetaValue<Long> width = new MetaValue<>(getMapWidth(mapNode,node,start.getValue()));

                additionalArguments.add(width);

                additionalArguments.add(start);
            } else if (function instanceof ReduceNode) {
                ReduceNode reduceNode = (ReduceNode) function;

                long start = getReductionStart(reduceNode,node);

                long width = getReductionWidth(reduceNode,node,start);

                long arity = getReductionArity(reduceNode);

                long depth = getReductionDepth(width,arity);

                ArrayList<Long> list = new ArrayList<>();

                list.add(width);
                list.add(arity);
                list.add(depth);
                list.add(start);

                MetaList<Long> meta = new MetaList<>(list);

                additionalArguments.add(meta);
            } else if (function instanceof DynamicProgrammingNode) {
                DynamicProgrammingNode dbNode = (DynamicProgrammingNode) function;

                // the time steps are the first additional argument for dynamic programming
                MetaValue<Long> timesteps = (MetaValue<Long>) node.getAdditionalArguments().get(0);

                node.getAdditionalArguments().remove(0);

                additionalArguments.add(timesteps);

                MetaList<Long> starts = new MetaList<>(getDynamicProgrammingStart(dbNode,node));

                MetaValue<Long> width = new MetaValue<>(getDynamicProgrammingWidth(dbNode,node,starts.getValues()));

                additionalArguments.add(width);

                additionalArguments.add(starts);

            } else if (function instanceof StencilNode) {
                StencilNode stencilNode = (StencilNode) function;

                MetaList<Long> starts = new MetaList<>(getStencilStarts(stencilNode,node));

                MetaList<Long> widths = new MetaList<>(getStencilWidths(stencilNode,node,starts.getValues()));

                additionalArguments.add(widths);
                additionalArguments.add(starts);

            }



            additionalArguments.addAll(node.getAdditionalArguments());

            node.setAdditionalArguments(additionalArguments);
        }

        /**
         * Visitor support functions.
         */
        private ExtendedShapeAPTVisitor realThis = this;

        @Override
        public ExtendedShapeAPTVisitor getRealThis() {
            return realThis;
        }

        public void setRealThis(ExtendedShapeAPTVisitor realThis) {
            this.realThis = realThis;
        }
    }

    private void generateAdditionalArguments() {
        AdditionalArgumentGenerator gen = new AdditionalArgumentGenerator();

        AbstractPatternTree.getFunctionTable().get("main").accept(gen);

    }

    /**
     * Returns the width (number of instances/iterations) of a stencil call.
     * @param function
     * @param call
     * @param initialOffsets
     * @return
     */
    private ArrayList<Long> getStencilWidths(StencilNode function, ParallelCallNode call, ArrayList<Long> initialOffsets) {
        ArrayList<Long> widths = new ArrayList<>();
        for (int dim = 0; dim < function.getDimension(); dim++) {
            long N = Long.MAX_VALUE;
            String currentRuleBase = "INDEX" + dim;
            for (int i = 0; i < call.getParameterCount() + 1; ++i) {
                Data argumentData;
                if (i == call.getParameterCount()) {
                    argumentData = function.getReturnElement();
                } else {
                    argumentData = function.getArgumentValues().get(i);
                }
                if (!(argumentData instanceof ArrayData)) {
                    continue;
                }
                ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
                for (DataAccess access : accesses) {
                    // Replace by correct access class.
                    if (access instanceof StencilDataAccess) {
                        StencilDataAccess stencilAccess = ((StencilDataAccess) access);
                        for (int j = 0; j < stencilAccess.getRuleBaseIndex().size(); j++) {
                            if (stencilAccess.getRuleBaseIndex().get(j).equals(currentRuleBase) && stencilAccess.getShiftOffsets().get(j) >= 0) {
                                N = Long.min((long) Math.floor((((ArrayData) argumentData).getShape().get(j) - 1 - stencilAccess.getShiftOffsets().get(j))/ (double) stencilAccess.getScalingFactors().get(j)), N);
                            }
                        }
                    }
                }
            }
            widths.add(N+1-initialOffsets.get(dim));
        }
        return widths;
    }

    /**
     * Returns the minimal initial offset to avoid negative indices for a stencil call.
     * @param function
     * @param call
     * @return
     */
    private ArrayList<Long> getStencilStarts(StencilNode function, ParallelCallNode call) {
        ArrayList<Long> starts = new ArrayList<>();
        for (int dim = 0; dim < function.getDimension(); dim++) {
            boolean hasSpecStart = false;
            long N = Long.MIN_VALUE;
            String currentRuleBase = "INDEX" + dim;
            for (int i = 0; i < call.getParameterCount() + 1; ++i) {
                Data argumentData;
                if (i == call.getParameterCount()) {
                    argumentData = function.getReturnElement();
                } else {
                    argumentData = function.getArgumentValues().get(i);
                }
                if (!(argumentData instanceof ArrayData)) {
                    continue;
                }
                ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
                for (DataAccess access : accesses) {
                    // Replace by correct access class.
                    if (access instanceof StencilDataAccess) {
                        StencilDataAccess stencilAccess = ((StencilDataAccess) access);
                        for (int j = 0; j < stencilAccess.getRuleBaseIndex().size(); j++) {
                            if (stencilAccess.getRuleBaseIndex().get(j).equals(currentRuleBase) && stencilAccess.getShiftOffsets().get(j) < 0) {
                                hasSpecStart = true;
                                long NOld = N;
                                N = Long.max((long) Math.floor((Math.abs(stencilAccess.getShiftOffsets().get(j))) / (double) stencilAccess.getScalingFactors().get(j)), N);
                                if ((Math.abs(stencilAccess.getShiftOffsets().get(j))) % stencilAccess.getScalingFactors().get(j) != 0 && (int) Math.floor((Math.abs(stencilAccess.getShiftOffsets().get(j))) / (double) stencilAccess.getScalingFactors().get(j)) >= NOld) {
                                    N++;
                                }
                            }
                        }
                    }
                }
            }
            if (!hasSpecStart) {
                N=0;
            }
            starts.add(N);
        }
        return starts;
    }

    /**
     *
     * Returns the width (number of instances/iterations) of a dynamic programming call.
     * This function may throw an error, if a data element is accessed based on time and the array is to small for a given number of time steps.
     * @param function
     * @param call
     * @param initialOffsets
     * @return
     */

    private long getDynamicProgrammingWidth(DynamicProgrammingNode function, ParallelCallNode call, ArrayList<Long> initialOffsets) {
        long N = Long.MAX_VALUE;
        boolean hasInternal = false;

        for (int i = 0; i < call.getParameterCount() + 1; ++i) {
            Data argumentData;
            if (i == call.getParameterCount()) {
                argumentData = function.getReturnElement();
            } else {
                argumentData = function.getArgumentValues().get(i);
            }
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof DynamicProgrammingDataAccess) {
                    DynamicProgrammingDataAccess dpAccess = ((DynamicProgrammingDataAccess) access);

                    if (dpAccess.getRuleBaseIndex().get(0).equals("INDEX0")) {
                        // get the number of time steps, since INDEX0 iterates over the time steps.
                        long diff = ((MetaValue<Integer>) call.getAdditionalArguments().get(0)).getValue() - (((ArrayData) argumentData).getShape().get(0) - dpAccess.getShiftOffsets().get(0) - initialOffsets.get(0));
                        if (diff > 0){
                            Log.error("Size of array " + argumentData.getIdentifier() + " to small for this number of time steps! Try " + diff + " more elements!");
                            throw new RuntimeException("Critical error!");
                        }
                    } else if (dpAccess.getRuleBaseIndex().get(0).equals("INDEX1") && dpAccess.getShiftOffsets().get(0) >= 0) {
                        N = Long.min((long) Math.floor(((ArrayData) argumentData).getShape().get(0) - 1 - dpAccess.getShiftOffsets().get(0)), N);
                        hasInternal = true;
                    }
                }
            }
        }
        if (hasInternal) {
            return N+1-initialOffsets.get(1);
        }
        return 1;
    }


    /**
     * Returns the minimal initial offset to avoid negative indices for a dynamic programming call.
     * @param function
     * @param call
     * @return
     */
    private ArrayList<Long> getDynamicProgrammingStart(DynamicProgrammingNode function, ParallelCallNode call) {

        ArrayList<Long> starts = new ArrayList<>();
        for (int dim = 0; dim < 2; dim++) {
            long N = Long.MIN_VALUE;
            boolean hasSpecStart = false;
            String currentRuleBase = "INDEX" + dim;
            for (int i = 0; i < call.getParameterCount() + 1; ++i) {
                Data argumentData;
                if (i == call.getParameterCount()) {
                    argumentData = function.getReturnElement();
                } else {
                    argumentData = function.getArgumentValues().get(i);
                }
                if (!(argumentData instanceof ArrayData)) {
                    continue;
                }

                ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
                for (DataAccess access : accesses) {
                    // Replace by correct access class.
                    if (access instanceof DynamicProgrammingDataAccess) {
                        DynamicProgrammingDataAccess dpAccess = ((DynamicProgrammingDataAccess) access);
                        if (dpAccess.getShiftOffsets().get(0) < 0 && dpAccess.getRuleBaseIndex().get(0).equals(currentRuleBase)) {
                            hasSpecStart = true;
                            N = Long.max(-dpAccess.getShiftOffsets().get(0), N);
                        }
                    }
                }
            }
            if (!hasSpecStart) {
                N=0;
            }
            starts.add(N);
        }
        return starts;
    }

    /**
     * Returns the width (number of instances/iterations) of a Map call.
     * @param function
     * @param call
     * @param initialOffset
     * @return
     */
    private long getMapWidth(MapNode function, ParallelCallNode call, long initialOffset) {
        long N = Long.MAX_VALUE;
        for (int i = 0; i < call.getParameterCount() + 1; ++i) {
            Data argumentData;
            if (i == call.getParameterCount()) {
                argumentData = function.getReturnElement();
            } else {
                argumentData = function.getArgumentValues().get(i);
            }
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() >= 0) {
                        N = Long.min((int) Math.floor((((ArrayData) argumentData).getShape().get(0) - 1 - mapAccess.getShiftOffset()) / (double) mapAccess.getScalingFactor()), N);
                    }
                }
            }
        }
        return N+1 - initialOffset;
    }

    /**
     * Returns the minimal initial offset to avoid negative indices for a map call.
     * @param function
     * @param call
     * @return
     */
    private long getMapStart(MapNode function, ParallelCallNode call) {
        long N = Long.MIN_VALUE;
        boolean hasSpecStart = false;
        for (int i = 0; i < call.getParameterCount() + 1; ++i) {
            Data argumentData;
            if (i == call.getParameterCount()) {
                argumentData = function.getReturnElement();
            } else {
                argumentData = function.getArgumentValues().get(i);
            }
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() < 0) {
                        long NOld = N;
                        hasSpecStart = true;
                        N = Long.max((int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()), N);
                        if (((Math.abs(mapAccess.getShiftOffset())) % mapAccess.getScalingFactor()) != 0 && (int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()) >= NOld) {
                            N++;
                        }
                    }
                }
            }
        }

        if (!hasSpecStart) {
            N=0;
        }
        return N ;
    }

    /**
     * Returns the width (number of instances/iterations) of a reduction call.
     * @param function
     * @param call
     * @param initialOffset
     * @return
     */
    private long getReductionWidth(ReduceNode function, ParallelCallNode call, long initialOffset) {
        long N = Long.MAX_VALUE;
        for (int i = 0; i < call.getParameterCount(); ++i) {
            Data argumentData = function.getArgumentValues().get(i);
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    N = Long.min((int) Math.floor((((ArrayData) argumentData).getShape().get(0) - 1 - mapAccess.getShiftOffset()) / (double) mapAccess.getScalingFactor()), N);
                }
            }
        }
        return N + 1 - initialOffset;
    }

    /**
     * Returns the minimal initial offset to avoid negative indices for a reduction call.
     * @param function
     * @param call
     * @return
     */
    private long getReductionStart(ReduceNode function, ParallelCallNode call) {
        long N = Long.MIN_VALUE;
        boolean hasSpecStart = false;
        for (int i = 0; i < call.getParameterCount(); ++i) {
            //Data inputData = call.getInputElements().get(i);
            Data argumentData = function.getArgumentValues().get(i);
            if (!(argumentData instanceof ArrayData)) {
                continue;
            }

            ArrayList<DataAccess> accesses = argumentData.getTrace().getDataAccesses();
            for (DataAccess access : accesses) {
                // Replace by correct access class.
                if (access instanceof MapDataAccess) {
                    MapDataAccess mapAccess = ((MapDataAccess) access);
                    if (mapAccess.getShiftOffset() < 0) {
                        long NOld = N;
                        hasSpecStart = true;
                        N = Long.max((int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()), N);
                        if (((Math.abs(mapAccess.getShiftOffset())) % mapAccess.getScalingFactor()) != 0 && (int) Math.floor(( Math.abs(mapAccess.getShiftOffset())) / (double) mapAccess.getScalingFactor()) >= NOld) {
                            N++;
                        }
                    }
                }
            }
        }

        if (!hasSpecStart) {
            N=0;
        }
        return N ;
    }

    /**
     * Calculates the depth of the reduction (The number of necessary synchronization steps).
     * @param width
     * @param arity
     * @return
     */
    private long getReductionDepth(long width, long arity) {
        long depth = (long) (Math.log10(width) / Math.log10(arity));
        return depth;
    }


    /**
     * Computes the arity of the reduction step from a given reduce node.
     * The arity is defined as the number of read data elements within the reduction step. e.g. "res += in1 * in2" has an arity of two, because in1 and in2 are both accessed once.
     * @param node
     * @return
     */
    private long getReductionArity(ReduceNode node) {
        if (node.getChildren().get(node.getChildren().size() - 1) instanceof SimpleExpressionBlockNode) {
            SimpleExpressionBlockNode expNode = (SimpleExpressionBlockNode) node.getChildren().get(node.getChildren().size() - 1);
            AssignmentExpression exp = (AssignmentExpression) expNode.getExpressionList().get(expNode.getExpressionList().size() - 1);
            return exp.getRhsExpression().getOperands().size();
        } else if (node.getChildren().get(node.getChildren().size() - 1) instanceof ComplexExpressionNode) {
            AssignmentExpression exp = (AssignmentExpression) ((ComplexExpressionNode) node.getChildren().get(node.getChildren().size() - 1)).getExpression();
            return exp.getRhsExpression().getOperands().size();
        } else {
            Log.error("Reduction not sufficiently defined!  " + node.getIdentifier());
            throw new RuntimeException("Critical error!");
        }
    }


    /************************************************
     *
     *
     * Helper Functions
     *
     *
     ************************************************/

    private int connectCall2Expression(IRLExpression exp, ArrayList<PatternNode> calls, int currentIndex) {

        OperationExpression operationExpression;

        int newIndex = currentIndex;

        if (exp instanceof OperationExpression) {
            operationExpression = (OperationExpression) exp;
        } else {
            operationExpression = ((AssignmentExpression) exp).getRhsExpression();
        }

        for (Data element: operationExpression.getOperands() ) {
            if (element instanceof FunctionInlineData && !element.getIdentifier().equals("Get_Size_Of_Vector")) {
                newIndex = connectCall2Expression(((FunctionInlineData) element).getCall(),calls,newIndex );
                ((CallNode) calls.get(newIndex)).setCallExpression((FunctionInlineData) element);
                newIndex++;
            }
        }
        return newIndex;
    }

    private int getDimensionality(ASTListType type, int n) {
        int result = n;
        if (type.getType() instanceof ASTListType) {
            result = getDimensionality((ASTListType) type.getType(), result + 1);
        }
        return result;
    }

    /**
     * Returns all call expressions within a given expression.
     * @param astExpression
     * @return
     */
    private ArrayList<ASTCallExpression> getFunctionCalls(ASTExpression astExpression) {
        return new helper().getCallExpressions(astExpression);
    }

    private ArrayList<ASTCallExpression> getFunctionCalls(ASTPatternCallStatement astExpression) {
        return new helper().getCallExpressions(astExpression);
    }

    /**
     * Helper class to get all call expressions within a given expression.
     */
    private class helper implements PatternDSLVisitor {
        private ArrayList<ASTCallExpression> callExpressions = new ArrayList<>();

        public helper() {
        }

        public ArrayList<ASTCallExpression> getCallExpressions(ASTExpression node) {
            node.accept(this);
            return callExpressions;
        }

        public ArrayList<ASTCallExpression> getCallExpressions(ASTPatternCallStatement node) {
            node.accept(this);
            return callExpressions;
        }

        @Override
        public void endVisit(ASTCallExpression node) {
            callExpressions.add(node);
        }

        /**
         * Visitor support functions.
         */
        private PatternDSLVisitor realThis = this;

        @Override
        public PatternDSLVisitor getRealThis() {
            return realThis;
        }

        @Override
        public void setRealThis(PatternDSLVisitor realThis) {
            this.realThis = realThis;
        }
    }


}
