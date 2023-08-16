package de.parallelpatterndsl.patterndsl.Preprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MainNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.helperLibrary.DeepCopyHelper;
import de.parallelpatterndsl.patterndsl.helperLibrary.PredefinedFunctions;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Functions with a parallel context can be called multiple times. Depending on the parameters the parallel patterns get different sizes.
 * To avoid mismatches these functions are deep copied and calls replaced so that parallel patterns don't have multiple additional parameters.
 */
public class CallStructure {

    private HashMap<FunctionNode, ArrayList<CallNode>> multiCallMap;

    private ArrayList<FunctionNode> order;

    private AbstractPatternTree APT;

    public CallStructure(AbstractPatternTree APT) {
        this.APT = APT;
        order = new ArrayList<>();
    }

    public void generate() {

        DeepCopyHelper.addScope(APT.getGlobalVariableTable());

        while (true) {
            //1. Fill multiCallMap with function nodes and all corresponding calls
            FunctionGatherer functionGatherer = new FunctionGatherer(APT.getRoot());
            multiCallMap = functionGatherer.generate();

            //2. Filter out all functions with only a single call
            ArrayList<FunctionNode> newOrder = new ArrayList<>();
            for (FunctionNode function : order) {
                if (multiCallMap.get(function).size() == 1) {
                    multiCallMap.remove(function);
                } else {
                    newOrder.add(function);
                }
            }
            order = new ArrayList<>();

            //3. Filter out all functions without parallel context
            for (FunctionNode function : newOrder) {
                if (!function.isHasParallelDescendants()) {
                    multiCallMap.remove(function);
                } else {
                    order.add(function);
                }
            }

            // Stop if finished
            if (multiCallMap.isEmpty()) {
                break;
            }

            FunctionNode function = order.get(0);
            //4. Create a copy of the call and function nodes and store them
            CallNode call = multiCallMap.get(function).get(0);

            String unrollIdentifier = RandomStringGenerator.getAlphaNumericString();

            FunctionNode copiedFunction = function.deepCopy();

            copiedFunction.addUnrollIdentifier(unrollIdentifier);
            AbstractPatternTree.getFunctionTable().put(copiedFunction.getIdentifier(), copiedFunction);

            //5. Replace call nodes with new copies linking only to a single function
            //6. Replace the names of data elements for correct replacements ( see addUnrollIdentifier())
            call.addUnrollIdentifier(unrollIdentifier);

            order = new ArrayList<>();
        }

        DeepCopyHelper.removeScope();
    }



    private class FunctionGatherer implements APTVisitor {

        private MainNode root;

        private HashMap<FunctionNode, ArrayList<CallNode>> multiCallMap = new HashMap<>();

        public FunctionGatherer(MainNode root) {
            this.root = root;
        }

        public HashMap<FunctionNode, ArrayList<CallNode>> generate() {

            root.accept(this.getRealThis());

            return multiCallMap;
        }


        @Override
        public void visit(CallNode node) {

            if (PredefinedFunctions.contains(node.getFunctionIdentifier())) {
                return;
            }

            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            if (multiCallMap.containsKey(function)) {
                multiCallMap.get(function).add(node);
            } else {
                order.add(function);
                ArrayList<CallNode> calls = new ArrayList<>();
                calls.add(node);
                multiCallMap.put(function, calls);
            }
        }

        @Override
        public void visit(ParallelCallNode node) {
            FunctionNode function = AbstractPatternTree.getFunctionTable().get(node.getFunctionIdentifier());
            if (multiCallMap.containsKey(function)) {
                multiCallMap.get(function).add(node);
            } else {
                order.add(function);
                ArrayList<CallNode> calls = new ArrayList<>();
                calls.add(node);
                multiCallMap.put(function, calls);
            }
        }
    }
}
