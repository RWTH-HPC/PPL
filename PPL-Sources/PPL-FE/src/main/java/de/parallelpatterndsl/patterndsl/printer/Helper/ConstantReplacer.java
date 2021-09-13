package de.parallelpatterndsl.patterndsl.printer.Helper;

import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._symboltable.ConstantSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.printer.AbstractExpressionPrinter;

import java.util.Optional;

/**
 * A helper class that traverses the AST and replaces all occurrences of a constant with its definition.
 */
public class ConstantReplacer implements PatternDSLVisitor {

    ASTModule module;

    AbstractExpressionPrinter<String> printer;

    public ConstantReplacer(ASTModule module, AbstractExpressionPrinter printer) {
        this.module = module;
        this.printer = printer;
    }

    public String replace() {
        module.accept(getRealThis());
        return "";
    }


    @Override
    public void visit(ASTNameExpression node) {
        String name = node.getName();
        Optional<ConstantSymbol> sym = node.getEnclosingScope().resolve(name, ConstantSymbol.KIND);
        if (!sym.isPresent()) {
            return;
        }
        ConstantSymbol constantSymbol = sym.get();
        if (constantSymbol.getNameValue().isPresent()) {
            constantSymbol.getNameValue().get().accept(getRealThis());
            node.setName(printer.doPrintExpression(constantSymbol.getNameValue().get()));
        } else if (constantSymbol.getListValue().isPresent()) {
            constantSymbol.getListValue().get().accept(getRealThis());
            node.setName(printer.doPrintExpression(constantSymbol.getListValue().get()));
        } else if (constantSymbol.getLiteralValue().isPresent()) {
            constantSymbol.getLiteralValue().get().accept(getRealThis());
            node.setName(printer.doPrintExpression(constantSymbol.getLiteralValue().get()));
        }
    }

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
