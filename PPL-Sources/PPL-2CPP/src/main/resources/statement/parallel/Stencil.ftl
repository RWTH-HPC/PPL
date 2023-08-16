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
        <#if func.sizeArgss() == 2>
            #pragma omp parallel for
            ${gh.generateStencilLoopHeaders(index, func)}
            ${includeArgs("statement/BlockStatement.ftl", function.getBlockStatement()) }
            ${gh.genStencilClosingBrackets(func)}
        </#if>
        <#if func.sizeArgss() == 3>
            #pragma omp parallel for num_threads($func.getArgs(0))
            ${gh.generateStencilLoopHeaders(index, func)}
            ${includeArgs("statement/BlockStatement.ftl", function.getBlockStatement()) }
            ${gh.genStencilClosingBrackets(func)
        </#if>
    </#if>
</#if>
