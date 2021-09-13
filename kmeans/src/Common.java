import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Common {

    public static final int THREAD_COUNT = 16;
    public static final int EXECUTIONS_COUNT = 10000;
    public static final long UNIT_DIVIDER = 1000; // microseconds

    static int populateData(List<String> positionals, HashMap<String, Double[]> data) throws IOException {//lettura del file per creare il set di dati da utilizzare
        BufferedReader csv = null;
        try {
            csv = new BufferedReader(new FileReader(positionals.get(0)));
        } catch (FileNotFoundException e) {
            System.err.println("given path does not exists");
            System.exit(1);
        }
        int n = parseData(csv, data);
        return n;
    }

    static boolean extractArguments(String[] args, List<String> positionals) {//gestire il flag -initClusters
        int c = 0;
        boolean initRandomClusters = false;
        while (c < args.length) {
            if ("-initClusters".equals(args[c])) {
                initRandomClusters = true;
            } else {
                positionals.add(args[c]);
            }
            c++;
        }
        return initRandomClusters;
    }

    private static int parseData(BufferedReader csv, HashMap<String, Double[]> data) throws IOException {
        String row = csv.readLine();
        double[] domainMax = null;
        int n = -1;
        while (row != null) {
            String[] rowArr = row.split(";");
            if (n == -1) {
                n = rowArr.length - 1;
                domainMax = new double[n];
                for (int i = 0; i < n; i++) {
                    domainMax[i] = Double.MIN_VALUE;
                }
            }
            Double[] dataArr = new Double[n];
            for (int i = 1; i < rowArr.length; i++) {
                dataArr[i - 1] = Double.parseDouble(rowArr[i]);
                if (dataArr[i - 1] > domainMax[i - 1]) {
                    domainMax[i - 1] = dataArr[i - 1];
                }
            }
            data.put(rowArr[0], dataArr);
            row = csv.readLine();
        }
        for (Double[] vector : data.values()) {
            for (int i = 0; i < n; i++) {
                vector[i] = vector[i] / domainMax[i]; // normalizza i dati -> tutto è adesso fra 0 e 1
            }
        }
        return n;
    }

    public static String formatTable(Object o) {
        StringBuilder sb = new StringBuilder();
        Formatter form = new Formatter(sb);
        if (o instanceof Object[]) {
            Object[] arr = (Object[]) o;
            List<List<String>> content = new ArrayList<>(arr.length);
            for (Object e : arr) {
                content.add(mapsObj(e));
            }
            for (int i = 0; i < arr.length; i++) {
                form.format("\t%d", i);
            }
            sb.append("\n");
            int j = 0;
            boolean next = true;
            String last = null;
            while (next) {
                next = false;
                for (int i = 0; i < arr.length; i++) {
                    List<String> col = content.get(i);
                    if (j < col.size()) {
                        String elem = col.get(j);
                        last = elem;
                        form.format("\t%s", elem);
                        if (j + 1 < col.size()) {
                            next = true;
                        }
                    } else {
                        if (last != null) { // questo aiuta
                            // ma questa funzione è ancora buggata...pace serve solo a stampare
                            int offset = last.length() / 8 + 1;
                            for (int h = 0; h < offset; h++) {
                                form.format("\t");
                            }
                        } else {
                            form.format("\t\t");
                        }
                    }
                }
                sb.append("\n");
                j++;
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static List<String> mapsObj(Object o) {
        List<String> lt = new LinkedList<>();
        if (o instanceof Iterable) {
            Iterable<Object> it = (Iterable<Object>) o;
            for (Object e : it) {
                lt.add(e.toString());
            }
        }
        return lt;
    }

    public static void printList(Iterable<String>[] S) {
        for (Iterable<String> strings : S) {
            for (String key : strings) {
                System.out.print(key + ", ");
            }
            System.out.println(".");
        }
    }

    public static void printMetrics(long[] metrics, int clustersN) {
        System.out.println("Executed " + EXECUTIONS_COUNT + " Kmeans, on " + THREAD_COUNT + " threads (if applicable)");
        System.out.println("Cluster number: " + clustersN);
        System.out.println("Total Kmeans execution count: " + EXECUTIONS_COUNT);
        System.out.println("Total execution time in µs : " + metrics[0] / UNIT_DIVIDER + ", average on kmean execution: " + metrics[1] / EXECUTIONS_COUNT / UNIT_DIVIDER);
        System.out.println("Norm execution time in µs : " + metrics[1] / UNIT_DIVIDER + ", average on kmean execution: " + metrics[1] / EXECUTIONS_COUNT / UNIT_DIVIDER);
        System.out.println("Means execution time in µs : " + metrics[2] / UNIT_DIVIDER + ", average on kmean execution: " + metrics[2] / EXECUTIONS_COUNT / UNIT_DIVIDER);
        System.out.println("Kmean execution time in µs : " + metrics[3] / UNIT_DIVIDER + ", average on kmean execution: " + metrics[3] / EXECUTIONS_COUNT / UNIT_DIVIDER);
    }
}
