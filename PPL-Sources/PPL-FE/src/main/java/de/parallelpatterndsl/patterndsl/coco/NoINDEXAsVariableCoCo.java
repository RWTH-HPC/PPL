package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTVariable;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTVariableCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._symboltable.VariableSymbol;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks if the access notation ([]) is used on Lists.
 */
public class NoINDEXAsVariableCoCo implements PatternDSLASTVariableCoCo {



    @Override
    public void check(ASTVariable node) {
        String name = node.getName();
        if (name.startsWith("INDEX")) {
            Log.error(node.get_SourcePositionStart() + " Do not use INDEX variables as a normal variables, they are a keyword in parallel patterns. Please rename this variable: " + node.getName());
        }

    }
}
