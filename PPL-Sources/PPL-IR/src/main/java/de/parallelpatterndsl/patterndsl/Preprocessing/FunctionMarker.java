package de.parallelpatterndsl.patterndsl.Preprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;

public class FunctionMarker implements APTVisitor {

    private AbstractPatternTree APT;

    public FunctionMarker(AbstractPatternTree APT) {
        this.APT = APT;
    }

    public void generate() {
        APT.getRoot().accept(this.getRealThis());
    }

    @Override
    public void visit(ParallelCallNode node) {
        if (!PredefinedFunctions.contains(node.getFunctionIdentifier())) {
            AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier()).setAvailableAfterInlining(true);
        }
    }

    @Override
    public void visit(CallNode node) {
        if (!PredefinedFunctions.contains(node.getFunctionIdentifier())) {
            AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier()).setAvailableAfterInlining(true);
        }
    }
}
