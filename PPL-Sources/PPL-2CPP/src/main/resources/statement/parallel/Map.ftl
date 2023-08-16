<#assign gh = glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->
${signature("func")}
<#assign functionOpt=gh.getFunction(func)>
<#if functionOpt.isPresent()>
    <#assign function=functionOpt.get().deepClone()>
    <#-- Replacement of variable names -->
    <#if function.isPresentFunctionParameter()>
        <#assign replacer = gh.getReplacer(function,func)>
        ${replacer.replace()}
        <#assign index = replacer.getIndex()>
        <#-- Code Generation depending on the ArgsList -->
        <#if func.sizeArgss() == 0>
            #pragma omp parallel for
            for( int INDEX_${index} = 0; INDEX_${index} < ${gh.printExpression(func.getArguments().getExpression(0))}.size(); INDEX_${index}++) {
            ${includeArgs("statement/BlockStatement.ftl", function.getBlockStatement()) } }
        </#if>
        <#if func.sizeArgss() == 1>
            #pragma omp parallel for
            for( int INDEX_${index} = 0; INDEX_${index} < ${gh.printExpression(func.getArgs(0))}; INDEX_${index}++) {
            ${includeArgs("statement/BlockStatement.ftl", function.getBlockStatement()) } }
        </#if>
        <#if func.sizeArgss() == 2>
            <#if gh.printExpression(func.getArgs(1)) == "0" || gh.printExpression(func.getArgs(1)) == "1">
                for( int INDEX_${index} = 0; INDEX_${index} < ${gh.printExpression(func.getArgs(0))}; INDEX_${index}++) {
                ${includeArgs("statement/BlockStatement.ftl", function.getBlockStatement()) } }
            <#else>
                #pragma omp parallel for num_threads(${gh.printExpression(func.getArgs(1))})
                for( int INDEX_${index} = 0; INDEX_${index} < ${gh.printExpression(func.getArgs(0))}; INDEX_${index}++) {
                ${includeArgs("statement/BlockStatement.ftl", function.getBlockStatement()) } }
            </#if>
        </#if>
    </#if>
</#if>
