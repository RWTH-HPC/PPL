package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTFunctionParameter;
import de.parallelpatterndsl.patterndsl._ast.ASTFunctionParameters;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReadOnlyHelper;
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;

/**
 * CoCo that checks, if parameters in pattern calls are only read from.
 */
public class ReadOnlyPatternParameterCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        ReadOnlyHelper readOnlyHelper = new ReadOnlyHelper(parameterNameList(node.getFunctionParameters()));
        if (!readOnlyHelper.isReadOnly(node)){
            Log.error(node.get_SourcePositionStart() + " Parameter change in function detected");
        }
    }

    private ArrayList<String> parameterNameList(ASTFunctionParameters parameters){
        ArrayList<String> res = new ArrayList<>();
        for (ASTFunctionParameter element: parameters.getFunctionParameterList()) {
            res.add(element.getName());
        }

        return res;
    }

}
