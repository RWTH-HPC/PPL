package de.parallelpatterndsl.patterndsl.AST2APTGenerator;

import de.monticore.literals.literals._ast.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.LiteralData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.PrimitiveDataTypes;
import de.parallelpatterndsl.patterndsl.printer.AbstractLiteralPrinter;

/**
 * Class that translates the Literal from the AST to Data elements in the APT.
 */
public class APTLiteralPrinter extends AbstractLiteralPrinter <Data> {

    private static APTLiteralPrinter printer;

    protected APTLiteralPrinter() { }

    protected static AbstractLiteralPrinter<Data> getInstance() {
        if (printer == null) {
            printer = new APTLiteralPrinter();
        }

        return printer;
    }

    public static Data printLiteral(ASTLiteral literal) {
        return APTLiteralPrinter.getInstance().doPrintLiteral(literal);
    }


    @Override
    protected LiteralData<Double> doPrintSignedDoubleLiteral(ASTSignedDoubleLiteral literal) {
        double value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.DOUBLE,value);
    }

    @Override
    protected LiteralData<Double> doPrintDoubleLiteral(ASTDoubleLiteral literal) {
        double value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.DOUBLE, value);
    }

    @Override
    protected LiteralData<Float> doPrintSignedFloatLiteral(ASTSignedFloatLiteral literal) {
        float value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.FLOAT,value);
    }

    @Override
    protected LiteralData<Float> doPrintFloatLiteral(ASTFloatLiteral literal) {
        float value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.FLOAT,value);
    }

    @Override
    protected LiteralData<Long> doPrintSignedLongliteral(ASTSignedLongLiteral literal) {
        long value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.INTEGER_64BIT,value);
    }

    @Override
    protected LiteralData<Long> doPrintLongLiteral(ASTLongLiteral literal) {
        long value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.INTEGER_64BIT,value);
    }

    @Override
    protected LiteralData<Integer> doPrintSignedIntLiteral(ASTSignedIntLiteral literal) {
        int value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.INTEGER_32BIT,value);
    }

    @Override
    protected LiteralData<Integer> doPrintIntLiteral(ASTIntLiteral literal) {
        int value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.INTEGER_32BIT,value);
    }

    @Override
    protected LiteralData<String> doPrintStringLiteral(ASTStringLiteral literal) {
        String value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.STRING,"\"" + value + "\"");
    }

    @Override
    protected LiteralData<Character> doPrintCharLiteral(ASTCharLiteral literal) {
        char value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.CHARACTER,value);
    }

    @Override
    protected LiteralData<Boolean> doPrintBooleanLiteral(ASTBooleanLiteral literal) {
        boolean value = literal.getValue();
        return new LiteralData<>("", PrimitiveDataTypes.BOOLEAN,value);
    }
}
