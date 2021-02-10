import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

public class KmeansND {
    public static void main(String[] args) throws Exception {
        BufferedReader csv = null;
        try {
            csv = new BufferedReader(new FileReader(args[0]));
        } catch (FileNotFoundException e) {
            System.err.println("given path does not exists");
            System.exit(1);
        }
        String row = csv.readLine();
        Map<String, Double[]> data = new HashMap<>();
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
        int clustersNumber = Integer.parseInt(args[1]); // args[0]
        double[][] initialMeans = new double[clustersNumber][n];
        Random r = new Random();
        int c = 0;
        double cosimoMin = Double.MAX_VALUE;
        List<String>[] G = null;
        while (c < 1000) {
            for (int i = 0; i < initialMeans.length; i++) { // array di dimensione clusterNumber contenente arrays di
                // dimensione n
                initialMeans[i] = r.doubles(n).toArray();
            }
            double[] totalNormAvg = new double[initialMeans.length];
            List<String>[] S = kmeans(data, initialMeans, totalNormAvg);
            List<String>[] F;
            do {
                initialMeans = mean(initialMeans, S, data);
                F = S;
                S = kmeans(data, initialMeans, totalNormAvg);
            } while (different(F, S));
            double cosimo2 = 0;
            for (int cosimo = 0; cosimo < totalNormAvg.length; cosimo++) {
                cosimo2 = cosimo2 + totalNormAvg[cosimo];
            }
            if (cosimoMin > cosimo2) {
                cosimoMin = cosimo2;
                G = S;
            }
            c++;
        }
        System.out.println(formatTable(G));
    }

    public static List<String>[] kmeans(Map<String, Double[]> data, double[][] initialMeans, double[] totalNormAvg) {
        List<String>[] S = new ArrayList[initialMeans.length];// dimensione clusterNumber
        for (int j = 0; j < S.length; j++) {
            S[j] = new ArrayList<>();
        }
        for (int h = 0; h < totalNormAvg.length; h++) {
            totalNormAvg[h] = 0;
        }
        int cont = 0;
        for (Entry<String, Double[]> entry : data.entrySet()) {
            int posMin = 0;

            double min = Double.MAX_VALUE;
            for (int h = 0; h < initialMeans.length; h++) {
                double norm = norm(entry.getValue(), initialMeans[h]);
                cont++;
                if (norm < min) {
                    min = norm;
                    posMin = h;
                }
            }
            S[posMin].add(entry.getKey());
            totalNormAvg[posMin] = totalNormAvg[posMin] + min;
        }
        // System.out.println("norm calcolata " + cont + " volte");
        for (int i = 0; i < totalNormAvg.length; i++) {
            if (S[i].size() > 0) {
                totalNormAvg[i] = totalNormAvg[i] / S[i].size();
            }
        }
        return S;
    }

    public static boolean different(List<String>[] F, List<String>[] S) {
        boolean no = false;
        for (int i = 0; i < F.length; i++) {
            if (!F[i].equals(S[i])) {
                no = true;
                break;
            }
        }
        return no;
    }

    public static void printList(Iterable<String>[] S) {
        for (Iterable<String> strings : S) {
            for (String key : strings) {
                System.out.print(key + ", ");
            }
            System.out.println(".");
        }
    }

    public static void printAnyThing(Object S) {
        if (S instanceof Object[]) {
            Object[] oArr = (Object[]) S;
            for (Object o : oArr) {
                printAnyThing(o);
                System.out.print(",");
            }
            System.out.println(".");
            System.out.println();
        } else if (S instanceof Iterable) {
            Iterable<Object> oIter = (Iterable<Object>) S;
            for (Object o : oIter) {
                printAnyThing(o);
                System.out.print(",");
            }
            System.out.println(".");
            System.out.println();
        } else {
            System.out.print(S.toString());
        }
    }

    public static String formatTable(Object o) {
        StringBuilder sb = new StringBuilder();
        Formatter form = new Formatter(sb);
        if (o instanceof Object[]) {
            Object[] arr = (Object[]) o;
            List<Object> lt = Arrays.asList(arr);
            List<List<String>> content = lt.stream().map(KmeansND::mapsObj).collect(Collectors.toList());
            for (int i = 0; i < arr.length; i++) {
                form.format("\t%d\t", i);
            }
            sb.append("\n");
            int j = 0;
            boolean next = true;
            while (next) {
                next = false;
                for (int i = 0; i < arr.length; i++) {
                    List<String> col = content.get(i);
                    if (j < col.size()) {
                        form.format("\t%s\t", col.get(j));
                        if (j + 1 < col.size()) {
                            next = true;
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

    public static double norm(Double[] a, double[] b) {
        double res = 0;
        for (int i = 0; i < a.length; i++) {
            res = res + Math.pow(a[i] - b[i], 2);
        }
        return res;
    }

    public static double[][] mean(double[][] initialMeans, List<String>[] S, Map<String, Double[]> data) {
        for (int i = 0; i < initialMeans.length; i++) {
            for (int j = 0; j < initialMeans[i].length; j++) {
                initialMeans[i][j] = 0;
                for (String key : S[i]) {
                    initialMeans[i][j] = initialMeans[i][j] + data.get(key)[j];
                }
                initialMeans[i][j] = initialMeans[i][j] / S[i].size();
            }
        }
        return initialMeans;
    }
}