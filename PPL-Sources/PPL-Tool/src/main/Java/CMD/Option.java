package CMD;

public class Option<T> {

    /**
     * The value given, if the Flag is not used.
     */
    private T defaultValue;

    /**
     * A description of the Flags use
     */
    private String Description;

    /**
     * True, iff the flag must always be given
     */
    private boolean required;

    /**
     * The short name of the Flag
     */
    private String shortFlag;

    /**
     * The long name of the Flag
     */
    private String longFlag;

    /**
     * True, iff the Flag is followed by an additional parameter
     */
    private boolean hasValue;

    /**
     * The value of the Flag after the arguments have been processed
     */
    private T value;

    private final Class<T> type;

    public Option(T defaultValue, String description, boolean required, String shortFlag, String longFlag, boolean hasValue, Class<T> type) {
        this.defaultValue = defaultValue;
        Description = description;
        this.required = required;
        this.shortFlag = shortFlag;
        this.longFlag = longFlag;
        this.hasValue = hasValue;
        this.value = defaultValue;
        this.type = type;
    }

    public T getDefaultValue() {
        return defaultValue;
    }

    public String getDescription() {
        return Description;
    }

    public boolean isRequired() {
        return required;
    }

    public String getShortFlag() {
        return shortFlag;
    }

    public String getLongFlag() {
        return longFlag;
    }

    public boolean isHasValue() {
        return hasValue;
    }

    public T getValue() {
        return value;
    }

    @SuppressWarnings("unchecked")
    public void setValue(String value) {
        if (type.isAssignableFrom(String.class)) {
            this.value = (T) value;
        } else if (type.isAssignableFrom(Integer.class)) {
            this.value = (T) Integer.valueOf(value);
        } else if (type.isAssignableFrom(Double.class)) {
            this.value = (T) Double.valueOf(value);
        } else if (type.isAssignableFrom(Boolean.class)) {
            this.value = (T) Boolean.valueOf(value);
        } else {
            throw new IllegalArgumentException("Bad type.");
        }
    }


}
