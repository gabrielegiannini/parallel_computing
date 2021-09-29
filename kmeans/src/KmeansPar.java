import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.BiConsumer;


public class KmeansPar {

    private final int clustersNumber;
    private final Map<String, Double[]> data;
    boolean initialized = false;
    private final double[][] centroids;
    private final DoubleAdder[][] nextCentroids;
    private Set<String>[] S;
    private Set<String>[] F;
    private final double[] totalNormAvg;
    private final DoubleAdder[] totalNormAcc;
    private final int n;

    public double[] getTotalNormAvg() {
        return totalNormAvg;
    }

    public KmeansPar(final Map<String, Double[]> data, int clustersNumber, final int n) throws IOException {//costruttore
        this.data = data;
        this.n = n;
        centroids = new double[clustersNumber][n];
        nextCentroids = new DoubleAdder[clustersNumber][n];
        for (int i = 0; i < clustersNumber; i++) {
            for (int j = 0; j < n; j++) {
                nextCentroids[i][j] = new DoubleAdder();
            }
        }
        this.clustersNumber = clustersNumber;
        totalNormAvg = new double[clustersNumber];
        totalNormAcc = new DoubleAdder[clustersNumber];

        for (int i = 0; i < clustersNumber; i++) {
            totalNormAcc[i] = new DoubleAdder();
        }
    }

    public void kmean(BiConsumer<Runnable, Integer> exec, Runnable start, Runnable wait) {
        S = new Set[centroids.length];// dimensione clusterNumber. S e' array di array list
        for (int j = 0; j < S.length; j++) {
            S[j] = Collections.synchronizedSet(new HashSet<>());
        }
        // array delle norme
        Arrays.fill(totalNormAvg, 0);

        int id = 0;
        if (!initialized) {
            for (final Entry<String, Double[]> entry : data.entrySet()) {
                exec.accept(() -> {
                    this.parallelizable(entry);
                }, id);
                id++;
            }
            initialized = true;
        }
        start.run();

        wait.run();

        for (int i = 0; i < totalNormAvg.length; i++) {
            if (S[i].size() > 0) {
                totalNormAvg[i] = totalNormAcc[i].sumThenReset() / S[i].size();
                for (int l = 0; l < n; l++) {
                    centroids[i][l] = nextCentroids[i][l].sumThenReset() / S[i].size();
                }
            }
        }
    }

    public void parallelizable(Entry<String, Double[]> entry) {
        int posMin = 0;
        double min = Double.MAX_VALUE;
        var vect = entry.getValue();
        for (int h = 0; h < centroids.length; h++) {
            double norm = norm(vect, centroids[h]);
            if (norm < min) {
                min = norm;
                posMin = h;
            }
        }
        //segna che il vettore appartiene al cluster posMin
        S[posMin].add(entry.getKey());
        totalNormAcc[posMin].add(min);

        //ora sappiamo a che cluster appartiene il vettore, aggiungiamolo alla somma per il calcolo del nuovo centroide di quel cluster
        for (int i = 0; i < n; i++) {
            nextCentroids[posMin][i].add(vect[i]);
        }
    }

    public double[] executeKMeans(BiConsumer<Runnable, Integer> exec, Runnable start, Runnable wait, long[] metrics) {
//        long startTime = System.nanoTime();
        long counter = 0;
        do {
            F = S;
            kmean(exec, start, wait);
            counter++;
        } while (notConverged());
//        long endTime = System.nanoTime();
//        metrics[3] += (endTime - startTime);
        metrics[4] += counter;
        return getTotalNormAvg();
    }

    public boolean notConverged() {
        if (F == null) return true;
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
        S = new Set[clustersNumber];
        List<String> keys = new ArrayList<>(data.keySet());
        for (int i = 0; i < clustersNumber; i++) {
            S[i] = new HashSet<>();
            S[i].add(keys.remove(r.nextInt(keys.size())));
        }
        // completa gli assegnamenti
        for (String k : keys) {
            S[r.nextInt(clustersNumber)].add(k);
        }

        for (int i = 0; i < centroids.length; i++) {
            for (int j = 0; j < centroids[i].length; j++) {
                centroids[i][j] = 0;
                for (String key : S[i]) {
                    centroids[i][j] = centroids[i][j] + data.get(key)[j];
                }
                centroids[i][j] = centroids[i][j] / S[i].size();
            }
        }
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
            centroids[i] = new double[n];
            Double[] mean = lt.get(index);
            for (int j = 0; j < mean.length; j++) {
                centroids[i][j] = mean[j];
            }
        }
    }

    public List<String>[] getClusters() {
        List[] ret = new List[S.length];
        for (int i = 0; i < S.length; i++) {
            ret[i] = new ArrayList<>(S[i]);
        }
        return ret;
    }

    public static double norm(Double[] a, double[] b) {
        double res = 0;
        for (int i = 0; i < a.length; i++) {
            res = res + Math.pow(a[i] - b[i], 2);
        }
        return res;
    }
}