import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class KmeansND {

    private final int clustersNumber;
    private Map<String, Double[]> data;
    private double[][] means;
    private List<String>[] S;
    private List<String>[] F;
    private double[] totalNormAvg;
    private int n;

    public double[] getTotalNormAvg() {
        return totalNormAvg;
    }

    public KmeansND(final Map<String, Double[]> data, int clustersNumber, final int n) throws IOException {
        this.data = data;
        this.n = n;
        means = new double[clustersNumber][n];
        this.clustersNumber = clustersNumber;
        totalNormAvg = new double[clustersNumber];
    }

    public void kmean() {
        S = new ArrayList[means.length];// dimensione clusterNumber
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
            for (int h = 0; h < means.length; h++) {
                double norm = norm(entry.getValue(), means[h]);
                cont++;
                if (norm < min) {
                    min = norm;
                    posMin = h;
                }
            }
            S[posMin].add(entry.getKey());
            totalNormAvg[posMin] = totalNormAvg[posMin] + min;
        }
        for (int i = 0; i < totalNormAvg.length; i++) {
            if (S[i].size() > 0) {
                totalNormAvg[i] = totalNormAvg[i] / S[i].size();
            }
        }
        // return S;
    }

    public void means() {
        for (int i = 0; i < means.length; i++) {
            for (int j = 0; j < means[i].length; j++) {
                means[i][j] = 0;
                for (String key : S[i]) {
                    means[i][j] = means[i][j] + data.get(key)[j];
                }
                means[i][j] = means[i][j] / S[i].size();
            }
        }
    }

    public double[] executeKMeans() {
        do {
            oneStepKMeans();
        } while (notConverged());
        return getTotalNormAvg();
    }

    public void oneStepKMeans() {
        kmean();
        means();
        F = S;
    }

    public boolean notConverged() {
        boolean no = false;
        for (int i = 0; i < F.length; i++) {
            if (!F[i].equals(S[i])) {
                no = true;
                break;
            }
        }
        return no;
    }

    public void initClusters() {
        Random r = new Random();
        S = new List[clustersNumber];
        List<String> keys = new ArrayList<>(data.keySet());
        for (int i = 0; i < clustersNumber; i++) {
            S[i] = new ArrayList<>();
            S[i].add(keys.remove(r.nextInt(keys.size())));
        }
        // completa gli assegnamenti
        for (String k : keys) {
            S[r.nextInt(clustersNumber)].add(k);
        }
        means(); // inizializza initialMeans
    }

    public void initMeans() {
        Random r = new Random();
        List<Double[]> lt = new ArrayList<>(data.values());
        Set<Integer> used = new HashSet<>();
        for (int i = 0; i < clustersNumber; i++) { // array di dimensione clusterNumber contenente arrays di
            // dimensione n

            /*
             * ok, pagina di wikipedia, sezione "Initialization methods"...non avevamo
             * capito nulla, ahah Quello che facevamo noi non era un metodo sano per
             * inizializzare i means! Su wiki sono spiegati 2 modi: selezionare k vettori a
             * caso tra i dati perché fungano da centroidi iniziali (è quelo che ho fatto io
             * sotto) oppure dividere casualmente tutti i dati fra i cluster e calcolare i
             * means da questa prima suddivisione casuale (wiki dice che i due metodi sono
             * preferibili in circostanze diverse, consideriamo di implementarli entrambi?
             * PS: implementati entrambi ;) )
             */
            int index = r.nextInt(lt.size());
            while (used.contains(index)) {
                index = r.nextInt(lt.size());
            }
            used.add(index);
            means[i] = new double[n];
            Double[] mean = lt.get(index);
            for (int j = 0; j < mean.length; j++) {
                means[i][j] = mean[j];
            }
        }
    }

    public List<String>[] getClusters() {
        return S;
    }

    // public void run(boolean initRandomClusters, List<double[]> totalNormAvg,
    // double minimunTotal,
    // List<List<String>[]> G) throws IOException {
    // KmeansND algorithm = new KmeansND(data, clustersNumber, n);
    // if (initRandomClusters) {
    // algorithm.initClusters();
    // } else {
    // algorithm.initMeans();
    // }
    // totalNormAvg.add(Integer.parseInt(this.getName()),
    // algorithm.executeKMeans());
    //
    // /*
    // * double total = 0; for (double avg : totalNormAvg) { total = total + avg; }
    // */
    //
    // G.add(Integer.parseInt(this.getName()), algorithm.getClusters());// prende il
    // nome del thread e ci mette il
    // // nuovo cluster
    // }

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