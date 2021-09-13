package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTDefinition;
import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._ast.ASTVariable;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTModuleCoCo;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReadOnlyHelperGlobalVars;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * CoCo that checks, if constants are only read from.
 */
public class ReadOnlyGlobalVarsCoCo implements PatternDSLASTModuleCoCo {

    @Override
    public void check(ASTModule node) {
        ArrayList<String> names = getGlobalVars(node);

        ReadOnlyHelperGlobalVars helper = new ReadOnlyHelperGlobalVars(names);
        if (!helper.isReadOnly(node)) {
            Log.error(node.get_SourcePositionStart() + " Change of global variable detected");
        }
    }

    private ArrayList<String> getGlobalVars(ASTModule node) {
        ArrayList<String> names = new ArrayList<>();

        for (ASTDefinition def: node.getDefinitionList().stream().filter(c -> c instanceof ASTVariable).collect(Collectors.toList()) ) {
            ASTVariable var = (ASTVariable) def;
            names.add(var.getName());
        }

        return names;
    }


}
