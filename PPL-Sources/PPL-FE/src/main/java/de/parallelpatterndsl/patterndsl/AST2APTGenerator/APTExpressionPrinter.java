package de.parallelpatterndsl.patterndsl.AST2APTGenerator;

import de.monticore.expressions.commonexpressions._ast.*;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import de.parallelpatterndsl.patterndsl.printer.AbstractExpressionPrinter;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Class implementing the expression printer to automatically generate the IRLExpressions from the AST.
 */
public class APTExpressionPrinter extends AbstractExpressionPrinter<IRLExpression> {

    /**
     * Reduced symbol table for variables.
     */
    private HashMap<String, Data> variableData;

    /**
     * Reduced symbol table for functions.
     */
    private HashMap<String, ASTFunction> astFunctionTable;

    public APTExpressionPrinter() {
    }



    public IRLExpression printExpression(ASTExpression expression, HashMap<String, Data> variableData, HashMap<String, ASTFunction> astFunctionTable) {
        this.variableData = variableData;
        this.astFunctionTable = astFunctionTable;

        return doPrintExpression(expression);
    }

    private AssignmentExpression generateAssignmentExpression(ASTExpression expression) {
        if (expression instanceof ASTAssignmentExpression) {
            return (AssignmentExpression) this.doPrintAssignmentExpression((ASTAssignmentExpression) expression);
        } else if (expression instanceof ASTSimpleAssignmentExpression) {
            return (AssignmentExpression) this.doPrintSimpleAssignmentExpression((ASTSimpleAssignmentExpression) expression);
        } else if (expression instanceof ASTAssignmentByIncreaseExpression) {
            return (AssignmentExpression) this.doPrintAssignmentByIncreaseExpression((ASTAssignmentByIncreaseExpression) expression);
        } else if (expression instanceof ASTAssignmentByDecreaseExpression) {
            return (AssignmentExpression) this.doPrintAssignmentByDecreaseExpression((ASTAssignmentByDecreaseExpression) expression);
        } else if (expression instanceof ASTAssignmentByMultiplyExpression) {
            return (AssignmentExpression) this.doPrintAssignmentByMultiplyExpression((ASTAssignmentByMultiplyExpression) expression);
        } else if (expression instanceof ASTDecrementExpression) {
            return (AssignmentExpression) this.doPrintDecrementExpression((ASTDecrementExpression) expression);
        } else if (expression instanceof ASTIncrementExpression) {
            return (AssignmentExpression) this.doPrintIncrementExpression((ASTIncrementExpression) expression);
        }
        throw new RuntimeException("Critical error!");
    }

    private OperationExpression generateOperationExpression(ASTExpression expression) {
        if (expression instanceof ASTListExpression) {
            return (OperationExpression) this.doPrintListExpression((ASTListExpression) expression);
        } else if (expression instanceof ASTBooleanAndOpExpression) {
            return (OperationExpression) this.doPrintBooleanAndOpExpression((ASTBooleanAndOpExpression) expression);
        } else if (expression instanceof ASTBooleanAndOpExpressionDiff) {
            return (OperationExpression) this.doPrintBooleanAndOpExpressionDiff((ASTBooleanAndOpExpressionDiff) expression);
        } else if (expression instanceof ASTBooleanOrOpExpression) {
            return (OperationExpression) this.doPrintBooleanOrOpExpression((ASTBooleanOrOpExpression) expression);
        } else if (expression instanceof ASTBooleanOrOpExpressionDiff) {
            return (OperationExpression) this.doPrintBooleanOrOpExpressionDiff((ASTBooleanOrOpExpressionDiff) expression);
        } else if (expression instanceof ASTLengthExpression) {
            return (OperationExpression) this.doPrintLengthExpression((ASTLengthExpression) expression);
        } else if (expression instanceof ASTInExpression) {
            return (OperationExpression) this.doPrintInExpression((ASTInExpression) expression);
        } else if (expression instanceof ASTIndexAccessExpression) {
            return (OperationExpression) this.doPrintIndexAccessExpression((ASTIndexAccessExpression) expression);
        } else if (expression instanceof ASTCallExpression) {
            return (OperationExpression) this.doPrintCallExpression((ASTCallExpression) expression);
        } else if (expression instanceof ASTNameExpression) {
            return (OperationExpression) this.doPrintNameExpression((ASTNameExpression) expression);
        } else if (expression instanceof ASTPlusExpression) {
            return (OperationExpression) this.doPrintPlusExpression((ASTPlusExpression) expression);
        } else if (expression instanceof ASTLitExpression) {
            return (OperationExpression) this.doPrintLitExpression((ASTLitExpression) expression);
        } else if (expression instanceof ASTBooleanNotExpression) {
            return (OperationExpression) this.doPrintBooleanNotExpression((ASTBooleanNotExpression) expression);
        } else if (expression instanceof ASTLogicalNotExpression) {
            return (OperationExpression) this.doPrintLogicalNotExpression((ASTLogicalNotExpression) expression);
        } else if (expression instanceof ASTMultExpression) {
            return (OperationExpression) this.doPrintMultExpression((ASTMultExpression) expression);
        } else if (expression instanceof ASTDivideExpression) {
            return (OperationExpression) this.doPrintDivideExpression((ASTDivideExpression) expression);
        } else if (expression instanceof ASTModuloExpression) {
            return (OperationExpression) this.doPrintModuloExpression((ASTModuloExpression) expression);
        } else if (expression instanceof ASTMinusExpression) {
            return (OperationExpression) this.doPrintMinusExpression((ASTMinusExpression) expression);
        } else if (expression instanceof ASTLessEqualExpression) {
            return (OperationExpression) this.doPrintLessEqualExpression((ASTLessEqualExpression) expression);
        } else if (expression instanceof ASTGreaterEqualExpression) {
            return (OperationExpression) this.doPrintGreaterEqualExpression((ASTGreaterEqualExpression) expression);
        } else if (expression instanceof ASTLessThanExpression) {
            return (OperationExpression) this.doPrintLessThanExpression((ASTLessThanExpression) expression);
        } else if (expression instanceof ASTGreaterThanExpression) {
            return (OperationExpression) this.doPrintGreaterThanExpression((ASTGreaterThanExpression) expression);
        } else if (expression instanceof ASTEqualsExpression) {
            return (OperationExpression) this.doPrintEqualsExpression((ASTEqualsExpression) expression);
        } else if (expression instanceof ASTNotEqualsExpression) {
            return (OperationExpression) this.doPrintNotEqualsExpression((ASTNotEqualsExpression) expression);
        } else if (expression instanceof ASTBracketExpression) {
            return (OperationExpression) this.doPrintBracketExpression((ASTBracketExpression) expression);
        } else if (expression instanceof  ASTRemainderExpressionDiff) {
            return (OperationExpression) this.doPrintRemainderExpressionDiff((ASTRemainderExpressionDiff) expression);
        } else if (expression instanceof ASTQualifiedNameExpression) {
            return (OperationExpression) this.doPrintQualifiedNameExpression((ASTQualifiedNameExpression) expression);
        } else if (expression instanceof ASTConditionalExpression) {
            return (OperationExpression) this.doPrintConditionalExpression((ASTConditionalExpression) expression);
        } else if (expression instanceof  ASTPrintExpression) {
            return (OperationExpression) this.doPrintPrintExpression((ASTPrintExpression) expression);
        } else if (expression instanceof ASTReadExpression) {
            return (OperationExpression) this.doPrintReadExpression((ASTReadExpression) expression);
        } else if (expression instanceof ASTBooleanNotOpExpressionDiff) {
            return (OperationExpression) this.doPrintBooleanNotExpressionDiff((ASTBooleanNotOpExpressionDiff) expression);
        }
        Log.error("Error in expression",expression.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }



    @Override
    protected IRLExpression doPrintWriteExpression(ASTWriteExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();

        IOData printCall = new IOData("write",PrimitiveDataTypes.COMPLEX_TYPE,true,true);

        operandList.add(printCall);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        operandList.add(APTLiteralPrinter.printLiteral(exp.getStringLiteral()));
        operatorList.add(Operator.COMMA);

        for (int i = 0; i < exp.getPrintElementList().size(); i++) {
            ASTPrintElement printElement = exp.getPrintElement(i);
            if (printElement.isPresentStringLiteral()) {
                operandList.add(APTLiteralPrinter.printLiteral(printElement.getStringLiteral()));
                if(i + 1 < exp.getPrintElementList().size()) {
                    operatorList.add(Operator.COMMA);
                }
            } else if (printElement.isPresentExpression()) {
                OperationExpression parameter = generateOperationExpression(printElement.getExpression());
                operandList.addAll(parameter.getOperands());
                operatorList.addAll(parameter.getOperators());
                if(i + 1 < exp.getPrintElementList().size()) {
                    operatorList.add(Operator.COMMA);
                }
            }
        }

        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        return new OperationExpression(operandList,operatorList);
    }



    @Override
    protected IRLExpression doPrintReadExpression(ASTReadExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();

        IOData printCall = new IOData("read",PrimitiveDataTypes.COMPLEX_TYPE,false,true);

        operandList.add(printCall);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        operandList.add(APTLiteralPrinter.printLiteral(exp.getStringLiteral()));

        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintAssignmentByIncreaseExpression(ASTAssignmentByIncreaseExpression exp) {
        ASTExpression left = exp.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data;
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
            return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()), Operator.PLUS_ASSIGNMENT);
        }
        Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    @Override
    protected IRLExpression doPrintRemainderExpressionDiff(ASTRemainderExpressionDiff exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.MODULO);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintDecrementExpression(ASTDecrementExpression exp) {
        ASTExpression left = exp.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data;
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
            return new AssignmentExpression(data,accessScheme,new OperationExpression(new ArrayList<>(), new ArrayList<>()),Operator.DECREMENT);
        }
        Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    @Override
    protected IRLExpression doPrintPrintExpression(ASTPrintExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();

        IOData printCall = new IOData("print",PrimitiveDataTypes.VOID,true, false);

        operandList.add(printCall);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        for (int i = 0; i < exp.getPrintElementList().size(); i++) {
            ASTPrintElement printElement = exp.getPrintElement(i);
            if (printElement.isPresentStringLiteral()) {
                operandList.add(APTLiteralPrinter.printLiteral(printElement.getStringLiteral()));
                if(i + 1 < exp.getPrintElementList().size()) {
                    operatorList.add(Operator.COMMA);
                }
            } else if (printElement.isPresentExpression()) {
                OperationExpression parameter = generateOperationExpression(printElement.getExpression());
                operandList.addAll(parameter.getOperands());
                operatorList.addAll(parameter.getOperators());
                if(i + 1 < exp.getPrintElementList().size()) {
                    operatorList.add(Operator.COMMA);
                }
            }
        }

        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintListExpression(ASTListExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();

        operatorList.add(Operator.LEFT_ARRAY_DEFINITION);

        for (int i = 0; i < exp.getExpressionList().size(); i++) {
            ASTExpression expression = exp.getExpression(i);
            OperationExpression arrayElement = generateOperationExpression(expression);

            operandList.addAll(arrayElement.getOperands());
            operatorList.addAll(arrayElement.getOperators());

            if ( i != exp.getExpressionList().size() - 1) {
                operatorList.add(Operator.COMMA);
            }
        }

        operatorList.add(Operator.RIGHT_ARRAY_DEFINITION);
        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintAssignmentExpression(ASTAssignmentExpression exp) {
        ASTExpression left = exp.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data;
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
            return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.ASSIGNMENT);
        }
        Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    @Override
    protected IRLExpression doPrintBooleanAndOpExpression(ASTBooleanAndOpExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.LOGICAL_AND);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintBooleanAndOpExpressionDiff(ASTBooleanAndOpExpressionDiff exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.LOGICAL_AND);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintBooleanOrOpExpression(ASTBooleanOrOpExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.LOGICAL_OR);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintBooleanOrOpExpressionDiff(ASTBooleanOrOpExpressionDiff exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.LOGICAL_OR);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintLengthExpression(ASTLengthExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();
        FunctionReturnData sizeCall = new FunctionReturnData("Get_Size_Of_Vector",PrimitiveDataTypes.INTEGER_32BIT);

        operandList.add(sizeCall);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        OperationExpression list = generateOperationExpression(exp.getExpression());
        operandList.addAll(list.getOperands());
        operatorList.addAll(list.getOperators());

        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        FunctionInlineData res = new FunctionInlineData("Get_Size_Of_Vector", PrimitiveDataTypes.INTEGER_32BIT, new OperationExpression(operandList,operatorList), 1);

        ArrayList<Data> operand = new ArrayList<>();
        operand.add(res);

        return new OperationExpression(operand, new ArrayList<>());
    }

    @Override
    protected IRLExpression doPrintInExpression(ASTInExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();
        FunctionReturnData existsCall = new FunctionReturnData("Element_Exists_In_Vector",PrimitiveDataTypes.BOOLEAN);

        operandList.add(existsCall);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        operandList.addAll(left.getOperands());
        operatorList.addAll(left.getOperators());

        operatorList.add(Operator.COMMA);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        FunctionInlineData res = new FunctionInlineData("Element_Exists_In_Vector", PrimitiveDataTypes.BOOLEAN, new OperationExpression(operandList,operatorList), 1);

         ArrayList<Data> operand = new ArrayList<>();
         operand.add(res);

        return new OperationExpression(operand, new ArrayList<>());
    }

    @Override
    protected IRLExpression doPrintIndexAccessExpression(ASTIndexAccessExpression exp) {

        ArrayList<OperationExpression> accesses = new ArrayList<>();

        ASTExpression expression = exp;
        // recursively iterate over all nested Index Access Expressions
        do {
            exp = (ASTIndexAccessExpression) expression;
            accesses.add(generateOperationExpression(exp.getIndex()));
            expression = exp.getExpression();
        } while(expression instanceof ASTIndexAccessExpression);

        // invert the order, ordering the Index Accesses by dimension (Descending)
        ArrayList<OperationExpression> accessesCorrectedOrder = new ArrayList<>();
        for (int i = accesses.size() - 1; i >= 0 ; i--) {
            accessesCorrectedOrder.add(accesses.get(i));
        }

        OperationExpression dataSource = generateOperationExpression(expression);
        ArrayList<Data> operandList = new ArrayList<>(dataSource.getOperands());
        ArrayList<Operator> operatorList = new ArrayList<>(dataSource.getOperators());

        for (OperationExpression operandExpression: accessesCorrectedOrder) {
            operatorList.add(Operator.LEFT_ARRAY_ACCESS);
            operandList.addAll(operandExpression.getOperands());
            operatorList.addAll((operandExpression.getOperators()));
            operatorList.add(Operator.RIGHT_ARRAY_ACCESS);
        }


        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintIncrementExpression(ASTIncrementExpression exp) {
            ASTExpression left = exp.getLeft();
            ArrayList<OperationExpression> accessScheme = new ArrayList<>();
            Data data;
            if (left instanceof ASTIndexAccessExpression) {
                accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
                while(left instanceof ASTIndexAccessExpression) {
                    left = ((ASTIndexAccessExpression) left).getExpression();
                }
            }
            if (left instanceof ASTNameExpression) {
                data = variableData.get(((ASTNameExpression) left).getName());
                return new AssignmentExpression(data,accessScheme,new OperationExpression(new ArrayList<>(), new ArrayList<>()),Operator.INCREMENT);
            }
            Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
            throw new RuntimeException("Critical error!");

    }

    //Creates a new variable for each function call.
    @Override
    protected IRLExpression doPrintCallExpression(ASTCallExpression callExpression) {
        ArrayList<Data> operandList = new ArrayList<>();
        ArrayList<Operator> operatorList = new ArrayList<>();

        String name = ((ASTNameExpression) callExpression.getExpression()).getName();
        // handle predefined functions
        PrimitiveDataTypes type;
        if (PredefinedFunctions.contains(name)) {
            type = PrimitiveDataTypes.COMPLEX_TYPE;
        } else {
            type = APTTypesPrinter.printType(astFunctionTable.get(name).getType());
        }

        FunctionReturnData returnData = new FunctionReturnData(name,type);

        String variableName = "inlineFunctionValue_" + RandomStringGenerator.getAlphaNumericString();

        // handle predefined functions
        int dimension;
        if (name.equals("init_List")) {
            dimension = ((ASTListExpression) callExpression.getArguments().getExpression(0)).sizeExpressions();
        } else if(PredefinedFunctions.contains(name)) {
            dimension = 1;
        } else {
            dimension = getDimensionality(astFunctionTable.get(name).getType(), 0);
        }

        operandList.add(returnData);
        operatorList.add(Operator.LEFT_CALL_PARENTHESIS);

        //generate arguments
        for (int i = 0; i < callExpression.getArguments().getExpressionList().size(); i++) {
            ASTExpression expression = callExpression.getArguments().getExpressionList().get(i);
            OperationExpression argumentExpression = generateOperationExpression(expression);
            operandList.addAll(argumentExpression.getOperands());
            operatorList.addAll(argumentExpression.getOperators());
            if (i != callExpression.getArguments().getExpressionList().size() - 1){
                operatorList.add(Operator.COMMA);
            }
        }
        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        // generate the inlining variable from the call expression
        FunctionInlineData callVariableData = new FunctionInlineData(variableName, type, new OperationExpression(operandList,operatorList), dimension);

        variableData.put(variableName, callVariableData);

        // return the inlining variable
        ArrayList<Data> resOperand = new ArrayList<>();
        resOperand.add(callVariableData);

        return new OperationExpression(resOperand, new ArrayList<>());
    }

    @Override
    protected IRLExpression doPrintNameExpression(ASTNameExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        operandList.add(variableData.get(exp.getName()));
        return new OperationExpression(operandList,new ArrayList<>());
    }

    @Override
    protected IRLExpression doPrintPlusExpression(ASTPlusExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.PLUS);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintQualifiedNameExpression(ASTQualifiedNameExpression exp) {
        //Expression got removed.
        return null;
    }

    @Override
    protected IRLExpression doPrintLogicalNotExpression(ASTLogicalNotExpression exp) {
        OperationExpression right = generateOperationExpression(exp.getExpression());
        ArrayList<Operator> operatorList =  new ArrayList<>();

        operatorList.add(Operator.LOGICAL_NOT);

        ArrayList<Data> operandList = new ArrayList<>(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }



    @Override
    protected IRLExpression doPrintBooleanNotExpression(ASTBooleanNotExpression exp) {
        OperationExpression right = generateOperationExpression(exp.getExpression());
        ArrayList<Operator> operatorList =  new ArrayList<>();

        operatorList.add(Operator.BITWISE_NOT);

        ArrayList<Data> operandList = new ArrayList<>(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintLitExpression(ASTLitExpression exp) {
        ArrayList<Data> operandList = new ArrayList<>();
        operandList.add(APTLiteralPrinter.printLiteral(exp.getLiteral()));
        return new OperationExpression(operandList,new ArrayList<>());
    }

    @Override
    protected IRLExpression doPrintAssignmentByMultiplyExpression(ASTAssignmentByMultiplyExpression exp) {
        ASTExpression left = exp.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data;
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
            return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.TIMES_ASSIGNMENT);
        }
        Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    @Override
    protected IRLExpression doPrintAssignmentByDecreaseExpression(ASTAssignmentByDecreaseExpression exp) {
        ASTExpression left = exp.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data;
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
            return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.MINUS_ASSIGNMENT);
        }
        Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    @Override
    protected IRLExpression doPrintBracketExpression(ASTBracketExpression exp) {
        OperationExpression right = generateOperationExpression(exp.getExpression());
        ArrayList<Operator> operatorList =  new ArrayList<>();

        operatorList.add(Operator.LEFT_PARENTHESIS);

        ArrayList<Data> operandList = new ArrayList<>(right.getOperands());
        operatorList.addAll(right.getOperators());

        operatorList.add(Operator.RIGHT_CALL_PARENTHESIS);

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintSimpleAssignmentExpression(ASTSimpleAssignmentExpression exp) {
        ASTExpression left = exp.getLeft();
        ArrayList<OperationExpression> accessScheme = new ArrayList<>();
        Data data;
        if (left instanceof ASTIndexAccessExpression) {
            accessScheme = getAccessScheme((ASTIndexAccessExpression) left,variableData,astFunctionTable);
            while(left instanceof ASTIndexAccessExpression) {
                left = ((ASTIndexAccessExpression) left).getExpression();
            }
        }
        if (left instanceof ASTNameExpression) {
            data = variableData.get(((ASTNameExpression) left).getName());
            if (exp.getOperator().equals("+=")) {
                return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.PLUS_ASSIGNMENT);
            } else if (exp.getOperator().equals("*=")) {
                return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.TIMES_ASSIGNMENT);
            } else if (exp.getOperator().equals("-=")) {
                return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.MINUS_ASSIGNMENT);
            }
            return new AssignmentExpression(data,accessScheme,generateOperationExpression(exp.getRight()),Operator.ASSIGNMENT);
        }
        Log.error("No assignment to variable possible, not a variable:" + left.toString() + " at:" + left.get_SourcePositionStart());
        throw new RuntimeException("Critical error!");
    }

    @Override
    protected IRLExpression doPrintConditionalExpression(ASTConditionalExpression exp) {
        // Expression got removed.
        return null;
    }

    @Override
    protected IRLExpression doPrintNotEqualsExpression(ASTNotEqualsExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.NOT_EQUAL);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintEqualsExpression(ASTEqualsExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.EQUAL);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintGreaterThanExpression(ASTGreaterThanExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.GREATER);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintLessThanExpression(ASTLessThanExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.LESS);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintGreaterEqualExpression(ASTGreaterEqualExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.GREATER_OR_EQUAL);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintLessEqualExpression(ASTLessEqualExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.LESS_OR_EQUAL);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintMinusExpression(ASTMinusExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.MINUS);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintModuloExpression(ASTModuloExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.MODULO);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintDivideExpression(ASTDivideExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.DIVIDE);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintMultExpression(ASTMultExpression exp) {
        OperationExpression left = generateOperationExpression(exp.getLeft());
        OperationExpression right = generateOperationExpression(exp.getRight());
        ArrayList<Data> operandList = left.getOperands();
        ArrayList<Operator> operatorList = left.getOperators();

        operatorList.add(Operator.MULTIPLICATION);

        operandList.addAll(right.getOperands());
        operatorList.addAll(right.getOperators());

        return new OperationExpression(operandList,operatorList);
    }

    @Override
    protected IRLExpression doPrintBooleanNotExpressionDiff(ASTBooleanNotOpExpressionDiff exp) {
            OperationExpression right = generateOperationExpression(exp.getExpression());
            ArrayList<Operator> operatorList =  new ArrayList<>();

            operatorList.add(Operator.BITWISE_NOT);

            ArrayList<Data> operandList = new ArrayList<>(right.getOperands());
            operatorList.addAll(right.getOperators());

            return new OperationExpression(operandList,operatorList);

    }

    /**
     * Returns the dimension of the given ASTList.
     * @param type
     * @param n
     * @return
     */
    public int getDimensionality(ASTType type, int n) {
        int result = n;
        if (type instanceof ASTListType) {
            result = getDimensionality(((ASTListType) type).getType(), result + 1);
        }
        return result;
    }

    /**
     * This function gives the index expressions within an Index-Access-Expression, if the result of an assignment expression is assigned to a partition of an Array.
     * @param exp
     * @return
     */
    public ArrayList<OperationExpression> getAccessScheme(ASTIndexAccessExpression exp, HashMap<String, Data> variableData, HashMap<String, ASTFunction> astFunctionTable) {
        ArrayList<OperationExpression> result = new ArrayList<>();
        ASTExpression expression = exp;

        this.variableData = variableData;
        this.astFunctionTable = astFunctionTable;
        // recursively iterate over all nested Index Access Expressions
        do {
            exp = (ASTIndexAccessExpression) expression;
            result.add(generateOperationExpression(exp.getIndex()));
            expression = exp.getExpression();
        } while(expression instanceof ASTIndexAccessExpression);

        // invert the order, ordering the Index Accesses by dimension (Descending)
        ArrayList<OperationExpression> resultCorrectedOrder = new ArrayList<>();
        for (int i = result.size() - 1; i >= 0 ; i--) {
            resultCorrectedOrder.add(result.get(i));
        }

        return resultCorrectedOrder;
    }
}
