import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;

public class KmeansND {
    public static void main(String[] args) throws Exception {
        BufferedReader csv = new BufferedReader(new FileReader("datasetProva.csv"));
        String row = csv.readLine();
        Map<String, Double[]> data = new HashMap<>();
        int n = 0;
        while (row != null) {
            String[] rowArr = row.split(";");
            n = rowArr.length - 1;
            Double[] dataArr = new Double[n];
            for (int i = 1; i < rowArr.length; i++) {
                dataArr[i - 1] = Double.parseDouble(rowArr[i]);
            }
            data.put(rowArr[0], dataArr);
            row = csv.readLine();
        }
        int clustersNumber = Integer.parseInt("10"); // args[0]
        double[][] initialMeans = new double[clustersNumber][n];
        Random r = new Random();
        for (int i = 0; i < initialMeans.length; i++) { // array di dimensione clusterNumber contenente arrays di
                                                        // dimensione n
            initialMeans[i] = r.doubles(n, -1, 11).toArray();
        }
        List<String>[] S = kmeans(data, initialMeans);
        List<String>[] F;
        do {
            initialMeans = mean(initialMeans, S, data);
            F = S;
            S = kmeans(data, initialMeans);
        } while (different(F, S));
        printList(S);
    }

    public static List<String>[] kmeans(Map<String, Double[]> data, double[][] initialMeans) {
        List<String>[] S = new ArrayList[initialMeans.length];// dimensione clusterNumber
        for (int j = 0; j < S.length; j++) {
            S[j] = new ArrayList<>();
        }
        for (Entry<String, Double[]> entry : data.entrySet()) {// questo dovrà farlo per le n dimensioni dell'input
            int posMin = 0;
            double min = Double.MAX_VALUE;// max value di double e questo sarà un vettore
            for (int h = 0; h < initialMeans.length; h++) {
                double norm = norm(entry.getValue(), initialMeans[h]);
                if (norm < min) {
                    min = norm;
                    posMin = h;
                }
            }
            S[posMin].add(entry.getKey());
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

    public static void printList(List<String>[] S) {
        for (int f = 0; f < S.length; f++) {
            for (String key : S[f]) {
                System.out.print(key + ", ");
            }
            System.out.println(".");
        }
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