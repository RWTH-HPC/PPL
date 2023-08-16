package Generator.HelperClasses;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ComplexExpressionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.SimpleExpressionBlockNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.printer.*;

import java.util.ArrayList;

public class ExpressionVisitor implements ExtendedShapeAPTVisitor {

    private ArrayList<String> prints = new ArrayList<>();

    public ArrayList<String> getPrints() {
        return prints;
    }

    @Override
    public void visit(ComplexExpressionNode node) {
        prints.add(CppExpressionPrinter.doPrintExpression(node.getExpression(), false, new ArrayList<>(), new ArrayList<>()));
    }

    @Override
    public void traverse(ParallelCallNode node) {
        for (int i = 1; i < node.getChildren().size(); i++) {
            PatternNode child = node.getChildren().get(i);
            child.accept(getRealThis());
        }
    }

    @Override
    public void visit(SimpleExpressionBlockNode node) {
        for (IRLExpression exp : node.getExpressionList()) {
            prints.add(CppExpressionPrinter.doPrintExpression(exp, false, new ArrayList<>(), new ArrayList<>()));
        }
    }
}
