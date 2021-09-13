package de.parallelpatterndsl.patterndsl.printer;


import de.monticore.literals.literals._ast.*;

/**
 * Class that implements a general literal printer. Usable for C++.
 */
public class LiteralPrinter extends AbstractLiteralPrinter<String>  {

    private static LiteralPrinter printer;

    protected LiteralPrinter() { }

    protected static AbstractLiteralPrinter<String> getInstance() {
        if (printer == null) {
            printer = new LiteralPrinter();
        }

        return printer;
    }

    public static String printLiteral(ASTLiteral literal) {
        return LiteralPrinter.getInstance().doPrintLiteral(literal);
    }


    @Override
    protected String doPrintSignedDoubleLiteral(ASTSignedDoubleLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintDoubleLiteral(ASTDoubleLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintSignedFloatLiteral(ASTSignedFloatLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintFloatLiteral(ASTFloatLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintSignedLongliteral(ASTSignedLongLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintLongLiteral(ASTLongLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintSignedIntLiteral(ASTSignedIntLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintIntLiteral(ASTIntLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }

    @Override
    protected String doPrintStringLiteral(ASTStringLiteral literal) {
        String erg = "\"" + literal.getValue() + "\"";
        return erg;
    }

    @Override
    protected String doPrintCharLiteral(ASTCharLiteral literal) {
        String erg = "\'" + literal.getValue() + "\'";
        return erg;
    }

    @Override
    protected String doPrintBooleanLiteral(ASTBooleanLiteral literal) {
        String erg = "" + literal.getValue();
        return erg;
    }
}
