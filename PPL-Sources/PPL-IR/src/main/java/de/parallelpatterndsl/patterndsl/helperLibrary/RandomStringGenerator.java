package de.parallelpatterndsl.patterndsl.helperLibrary;

import java.util.ArrayList;

/**
 * A helper class that generates random Strings.
 */
public class RandomStringGenerator {

    private static int n = 10;
    // list of already used random strings to avoid duplicates
    private static ArrayList<String> usedStrings = new ArrayList<String>();
    // function to generate a random string of length n
    public static String getAlphaNumericString()
    {




        // chose a Character random from this String
        String AlphaNumericString = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                + "0123456789"
                + "abcdefghijklmnopqrstuvxyz";

        // create StringBuffer size of AlphaNumericString
        StringBuilder sb = new StringBuilder(n);

        for (int i = 0; i < n; i++) {

            // generate a random number between
            // 0 to AlphaNumericString variable length
            int index
                    = (int)(AlphaNumericString.length()
                    * Math.random());

            // add Character one by one in end of sb
            sb.append(AlphaNumericString
                    .charAt(index));
        }
        String result = sb.toString();

        if (usedStrings.contains(result)) {
            result = getAlphaNumericString();
        } else {
            usedStrings.add(result);
        }
        return result;
    }

    public static void setN(int n) {
        RandomStringGenerator.n = n;
    }
}
