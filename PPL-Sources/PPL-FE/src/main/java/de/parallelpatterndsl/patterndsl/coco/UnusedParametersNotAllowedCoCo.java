package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTFunctionParameter;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks if the access notation ([]) is used in pattern calls.
 */
public class UnusedParametersNotAllowedCoCo implements PatternDSLASTFunctionCoCo {



    @Override
    public void check(ASTFunction node) {
        Helper helper = new Helper();
        helper.help(node);
    }

    private static class Helper implements PatternDSLVisitor {
        private boolean isUsed = false;

        private String variableName = "";

        @Override
        public void visit(ASTNameExpression node) {
            if (node.getName().equals(variableName)) {
                isUsed = true;
            }
        }

        public void help(ASTFunction node) {
            if (node.getPatternType().isPresentSerial()) {
                return;
            }

            for (ASTFunctionParameter parameter: node.getFunctionParameters().getFunctionParameterList() ) {
                isUsed = false;
                variableName = parameter.getName();
                node.accept(getRealThis());
                if (!isUsed) {
                    Log.error(node.get_SourcePositionStart() + " Parameter " + variableName + " in function " + node.getName() + " not used.");
                }
            }
        }
    }
}
