package CMD;

public class HandleComandLine {

    public static void parse(String[] argv) {

        for (int i = 0; i < argv.length; i++) {
            String argument = argv[i];

            ToolOptions currentFlag = ToolOptions.HELP;

            if (argument.startsWith("--")) {
                ToolOptions testing;
                if (argument.contains("=")) {
                    testing = ToolOptions.Flags.get(argument.split("=")[0]);
                } else {
                    testing = ToolOptions.Flags.get(argument);
                }
                if (testing != null) {
                    currentFlag = testing;
                }
            } else if (argument.startsWith("-")) {
                ToolOptions testing = ToolOptions.Flags.get(argument);
                if (testing != null) {
                    currentFlag = testing;
                }
            } else {
                System.out.println("Error: invalid Argument: " + argument);
                ToolOptions.getHelpMessage();
            }

            if (currentFlag == ToolOptions.HELP) {
                ToolOptions.getHelpMessage();
            } else if (currentFlag == ToolOptions.VERSION) {
                ToolOptions.getVersion();
            }

            Option opt = ToolOptions.options.get(currentFlag);
            if (opt.isHasValue()) {
                if (argument.startsWith("--")) {
                    opt.setValue(argument.split("=", 2)[1]);
                } else if (argument.startsWith("-")) {
                    if (argv.length == i + 1) {
                        System.out.println("Error: Missing parameter for: " + argument);
                        ToolOptions.getHelpMessage();
                    }
                    opt.setValue(argv[i+1]);
                    i++;
                } else {
                    System.out.println("Error: invalid Argument: " + argument);
                    ToolOptions.getHelpMessage();
                }
            } else {
                opt.setValue("true");
            }
        }

        for (Option opt: ToolOptions.options.values() ) {
            if (opt.isRequired()) {
                if (opt.getDefaultValue().equals(opt.getValue())) {
                    System.out.println("Error: required flag missing: " + opt.getShortFlag() + ", " + opt.getLongFlag());
                    ToolOptions.getHelpMessage();
                }
            }
        }

    }
}
