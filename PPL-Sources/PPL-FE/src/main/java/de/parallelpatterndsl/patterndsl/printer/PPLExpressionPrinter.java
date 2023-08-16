package de.parallelpatterndsl.patterndsl.printer;

/**
import de.monticore.expressions.commonexpressions._ast.*;
import de.monticore.expressions.expressionsbasis._ast.ASTExpression;

 **/
import de.parallelpatterndsl.patterndsl._ast.*;

/**
 * Class that implements the expression printer for C++ code.
 */
public class PPLExpressionPrinter extends AbstractExpressionPrinter <String> {

    private static PPLExpressionPrinter printer;

    private PPLExpressionPrinter() { }

    @Override
    protected String doPrintAssignmentByBitwiseOrExpression(ASTAssignmentByBitwiseOrExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " |= ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintAssignmentByBitwiseAndExpression(ASTAssignmentByBitwiseAndExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " &= ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBitwiseOrExpression(ASTBitwiseOrExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " | ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBitwiseNotExpression(ASTBitwiseNotExpression exp) {
        String erg = "";
        erg += " ~";
        erg += this.doPrintExpression(exp.getExpression());
        return erg;
    }

    @Override
    protected String doPrintBitwiseAndExpression(ASTBitwiseAndExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " & ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintWriteExpression(ASTWriteExpression exp) {
        String erg = "write(";
        erg += LiteralPrinter.printLiteral(exp.getStringLiteral());
        erg += ", ";
        for (int i = 0; i < exp.sizePrintElements(); i++) {
            ASTPrintElement akt = exp.getPrintElement(i);
            if (akt.isPresentExpression()){
                erg += this.doPrintExpression( akt.getExpression());
            } else if (akt.isPresentStringLiteral()){
                erg += LiteralPrinter.printLiteral( akt.getStringLiteral());
            }
            if(i < exp.sizePrintElements() - 1) {
                erg += ", ";
            }
        }
        erg += ")";
        return erg;
    }

    @Override
    protected String doPrintReadExpression(ASTReadExpression exp) {
        String erg = "read(";
        erg += LiteralPrinter.printLiteral(exp.getStringLiteral());
        erg += ")";
        return erg;
    }

    public static AbstractExpressionPrinter<String> getInstance() {
        if (printer == null) {
            printer = new PPLExpressionPrinter();
        }

        return printer;
    }

    public static String printExpression(ASTExpression expression) {
        return PPLExpressionPrinter.getInstance().doPrintExpression(expression);
    }


    @Override
    protected String doPrintRemainderExpressionDiff(ASTRemainderExpressionDiff exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " % ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintDecrementExpression(ASTDecrementExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += "--";
        return erg;
    }

    @Override
    protected String doPrintPrintExpression(ASTPrintExpression exp) {
        String erg = "print(";
        for (int i = 0; i < exp.sizePrintElements(); i++) {
            ASTPrintElement akt = exp.getPrintElement(i);
            if (akt.isPresentExpression()){
                erg += this.doPrintExpression( akt.getExpression());
            } else if (akt.isPresentStringLiteral()){
                erg += LiteralPrinter.printLiteral( akt.getStringLiteral());
            }
            if(i < exp.sizePrintElements() - 1) {
                erg += ", ";
            }
        }
        erg += ")";
        return erg;
    }

    @Override
    protected String doPrintListExpression(ASTListExpression exp) {
        String erg = "{";
        for (int i = 0; i < exp.sizeExpressions(); i++) {
            erg += this.doPrintExpression(exp.getExpression(i));
            if(i < exp.sizeExpressions() - 1) {
                erg += ", ";
            }
        }
        erg += "}";
        return erg;
    }

    @Override
    protected String doPrintAssignmentExpression(ASTAssignmentExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " = ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBooleanAndOpExpression(ASTBooleanAndOpExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " && ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBooleanAndOpExpressionDiff(ASTBooleanAndOpExpressionDiff exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " && ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBooleanOrOpExpression(ASTBooleanOrOpExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " || ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBooleanOrOpExpressionDiff(ASTBooleanOrOpExpressionDiff exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += " || ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintLengthExpression(ASTLengthExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getExpression());
        erg += ".size()";
        return erg;
    }

    @Override
    protected String doPrintInExpression(ASTInExpression exp) {
        String erg = "check_in(";
        erg += this.doPrintExpression(exp.getRight());
        erg += ", ";
        erg += this.doPrintExpression(exp.getLeft());
        erg += ")";
        return erg;
    }

    @Override
    protected String doPrintIndexAccessExpression(ASTIndexAccessExpression exp) {
        String erg = "";
        if(exp == null){System.out.println(5);}
        erg += this.doPrintExpression(exp.getIndexAccess());
        erg += "[";
        erg += this.doPrintExpression(exp.getIndex());
        erg += "]";
        return erg;
    }

    @Override
    protected String doPrintIncrementExpression(ASTIncrementExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getLeft());
        erg += "++";
        return erg;
    }

    @Override
    protected String doPrintCallExpression(ASTCallExpression exp) {
        String erg = "";
        erg += this.doPrintExpression(exp.getCall());
        erg += "(";
        for (int i = 0; i < exp.getArguments().sizeExpressions(); i++) {
            erg += this.doPrintExpression(exp.getArguments().getExpression(i));
            if (i < exp.getArguments().sizeExpressions() - 1) {
                erg += ", ";
            }
        }
        erg += ")";
        return erg;
    }

    @Override
    protected String doPrintNameExpression(ASTNameExpression exp) {
        return exp.getName();
    }

    @Override
    protected String doPrintPlusExpression(ASTPlusExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " + ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }
    /**
    @Override
    protected String doPrintQualifiedNameExpression(ASTQualifiedNameExpression exp) {
        String erg = this.doPrintExpression(exp.getExpression());
        erg += ".get_";
        erg += exp.getName();
        erg += "()";
        return erg;
    }
    **/
    @Override
    protected String doPrintLogicalNotExpression(ASTLogicalNotExpression exp) {
        return "!" + this.doPrintExpression(exp.getExpression());
    }

    @Override
    protected String doPrintBooleanNotExpression(ASTBooleanNotExpression exp) {
        return "~" + this.doPrintExpression(exp.getExpression());
    }

    @Override
    protected String doPrintLiteralExpression(ASTLiteralExpression exp) {
        return LiteralPrinter.printLiteral(exp.getLiteral());
    }



    @Override
    protected String doPrintAssignmentByMultiplyExpression(ASTAssignmentByMultiplyExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " *= ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintAssignmentByDecreaseExpression(ASTAssignmentByDecreaseExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " -= ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintAssignmentByIncreaseExpression(ASTAssignmentByIncreaseExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " += ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBracketExpression(ASTBracketExpression exp) {
        String erg = "(";
        erg += this.doPrintExpression(exp.getExpression());
        erg += ")";
        return erg;
    }
    /**
    @Override
    protected String doPrintSimpleAssignmentExpression(ASTSimpleAssignmentExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " += ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }
    **/
    @Override
    protected String doPrintConditionalExpression(ASTConditionalExpression exp) {
        String erg = this.doPrintExpression(exp.getCondition());
        erg += " ? ";
        erg += this.doPrintExpression(exp.getTrueExpression());
        erg += " : ";
        erg += this.doPrintExpression(exp.getFalseExpression());
        return erg;
    }

    @Override
    protected String doPrintNotEqualsExpression(ASTNotEqualsExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " != ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintEqualsExpression(ASTEqualsExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " == ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintGreaterThanExpression(ASTGreaterThanExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " > ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintLessThanExpression(ASTLessThanExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " < ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintGreaterEqualExpression(ASTGreaterEqualExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " >= ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintLessEqualExpression(ASTLessEqualExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " <= ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintMinusExpression(ASTMinusExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " - ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintModuloExpression(ASTModuloExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " % ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintDivideExpression(ASTDivideExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " / ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintMultExpression(ASTMultExpression exp) {
        String erg = this.doPrintExpression(exp.getLeft());
        erg += " * ";
        erg += this.doPrintExpression(exp.getRight());
        return erg;
    }

    @Override
    protected String doPrintBooleanNotExpressionDiff(ASTBooleanNotOpExpressionDiff exp) {
        return "~" + this.doPrintExpression(exp.getExpression());
    }
}
