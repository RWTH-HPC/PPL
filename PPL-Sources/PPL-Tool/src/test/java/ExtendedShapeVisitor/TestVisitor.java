package ExtendedShapeVisitor;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.ArrayData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;

import java.util.ArrayList;

public class TestVisitor implements ExtendedShapeAPTVisitor {

    @Override
    public void visit(ParallelCallNode node) {
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());

        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData) {
                ArrayList<Integer> shape = ((ArrayData) parameter).getShape();
                int dim = shape.size();
            }
        }
    }

    @Override
    public void visit(CallNode node) {
        FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());

        for (Data parameter: function.getArgumentValues() ) {
            if (parameter instanceof ArrayData) {
                ArrayList<Integer> shape = ((ArrayData) parameter).getShape();
                int dim = shape.size();
            }
        }
    }
}
