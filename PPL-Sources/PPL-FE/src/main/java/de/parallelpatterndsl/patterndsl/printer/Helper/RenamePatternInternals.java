package de.parallelpatterndsl.patterndsl.printer.Helper;
/**
import de.monticore.expressions.commonexpressions._ast.ASTNameExpression;
 **/
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.monticore.symboltable.GlobalScope;
import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTPatternCallStatement;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;
import de.parallelpatterndsl.patterndsl.printer.AbstractExpressionPrinter;

import java.util.Optional;

/**
 * A Class that replaces occurrences of INDEX to INDEX_$RandomString$, and replaces the parameters with the arguments.
 * Only for parallel patterns.
 */
public class RenamePatternInternals implements PatternDSLVisitor {

    ASTFunction Function;
    ASTPatternCallStatement Call;
    GlobalScope symbolTable;
    String Index;
    String resName = "";
    AbstractExpressionPrinter<String> printer;

    boolean replacing;

    public RenamePatternInternals(GlobalScope symbolTable, ASTFunction func, ASTPatternCallStatement call, AbstractExpressionPrinter printer) {
        this.symbolTable = symbolTable;
        this.Function = func;
        this.Call = call;
        this.Index = randIndex();
        this.printer = printer;
    }

    public String getIndex() {
        return Index;
    }

    public String replace() {
        replacing = true;
        Function.accept(getRealThis());
        return "";
    }

    public String replace(String resName) {
        this.resName = resName;
        replacing = true;
        Function.accept(getRealThis());
        return "";
    }


    public String randIndex(){
        StringBuilder sb = new StringBuilder();
        sb.append(RandomStringGenerator.getAlphaNumericString());
        return sb.toString();
    }

    @Override
    public void visit(ASTNameExpression node) {
        String name = node.getName();
        Optional<FunctionSymbol> sym = symbolTable.resolve(name, FunctionSymbol.KIND);
        if (sym.isPresent() && !replacing) {
            return;
        }
        if(name.equals(Function.getFunctionParameter().getName())){
            if (!resName.equals("")) {
                node.setName(resName);
            } else {
                node.setName(printer.doPrintExpression(Call.getLeft()));
            }
        }
        if(name.startsWith("INDEX")){
            node.setName(node.getName() + "_" + Index);
        }
        for (int i = 0; i < Function.getFunctionParameters().sizeFunctionParameters(); i++) {
            if(name.equals(Function.getFunctionParameters().getFunctionParameter(i).getName())) {
                node.setName(printer.doPrintExpression(Call.getArguments().getExpression(i)));
            }
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
