import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class Kmeans {

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

    public Kmeans(final Map<String, Double[]> data, int clustersNumber, final int n) throws IOException {
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
        for (Entry<String, Double[]> entry : data.entrySet()) {
            int posMin = 0;

            double min = Double.MAX_VALUE;
            for (int h = 0; h < means.length; h++) {
                double norm = norm(entry.getValue(), means[h]);
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

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        List<String> positionals = new LinkedList<String>();

        boolean initRandomClusters = Common.extractArguments(args, positionals);
        HashMap<String, Double[]> data = new HashMap<>();
        int n = Common.populateData(positionals, data);
        int clustersNumber = Integer.parseInt(positionals.get(1)); // args[0]
        double minimunTotal = Double.MAX_VALUE;
        List<String>[] G = null;
        int c = 0;
        while (c < Common.EXECUTIONS_COUNT) {
            Kmeans algorithm = new Kmeans(data, clustersNumber, n);
            if (initRandomClusters) {
                algorithm.initClusters();
            } else {
                algorithm.initMeans();
            }
            double[] totalNormAvg = algorithm.executeKMeans();

            double total = 0;
            for (double avg : totalNormAvg) {
                total = total + avg;
            }
            if (minimunTotal > total) {
                minimunTotal = total;
                G = algorithm.getClusters();
            }
            c++;
        }
        System.out.println(Common.formatTable(G));
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in milliseconds : " + timeElapsed / 1000000);
    }

    public static double norm(Double[] a, double[] b) {
        double res = 0;
        for (int i = 0; i < a.length; i++) {
            res = res + Math.pow(a[i] - b[i], 2);
        }
        return res;
    }
}