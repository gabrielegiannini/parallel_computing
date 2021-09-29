import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;
import java.util.function.BiConsumer;


public class Kmeans {

    private final int clustersNumber;
    private final Map<String, Double[]> data;
    private final double[][] centroids;
    private final double[][] nextCentroids;
    private Set<String>[] S;
    private Set<String>[] F;
    private final double[] totalNormAvg;
    private final int n;

    public double[] getTotalNormAvg() {
        return totalNormAvg;
    }

    public Kmeans(final Map<String, Double[]> data, int clustersNumber, final int n) throws IOException {//costruttore
        this.data = data;
        this.n = n;
        centroids = new double[clustersNumber][n];
        nextCentroids = new double[clustersNumber][n];
        for (int i = 0; i < clustersNumber; i++) {
            for (int j = 0; j < n; j++) {
                nextCentroids[i][j] = 0.0;
            }
        }
        this.clustersNumber = clustersNumber;
        totalNormAvg = new double[clustersNumber];
    }

    public void kmean() {
        S = new Set[centroids.length];// dimensione clusterNumber. S e' array di array list
        for (int j = 0; j < S.length; j++) {
            S[j] = Collections.synchronizedSet(new HashSet<>());
        }
        // array delle norme
        Arrays.fill(totalNormAvg, 0);
        for(int i = 0; i < clustersNumber;i++){
            Arrays.fill(nextCentroids[i], 0.0);
        }

        for (final Entry<String, Double[]> entry : data.entrySet()) {
            parallelizable(entry);
        }

        for (int i = 0; i < totalNormAvg.length; i++) {
            if (S[i].size() > 0) {
                totalNormAvg[i] = totalNormAvg[i] / S[i].size();
                for (int l = 0; l < n; l++) {
                    centroids[i][l] = nextCentroids[i][l] / S[i].size();
                }
            }
        }
    }

    public void parallelizable(Entry<String, Double[]> entry) {
//        long trueIndex = blockOffset + blockIdx.x * 1024 + threadIdx.x;
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
        totalNormAvg[posMin] += min;

        //ora sappiamo a che cluster appartiene il vettore, aggiungiamolo alla somma per il calcolo del nuovo centroide di quel cluster
        for (int i = 0; i < n; i++) {
            nextCentroids[posMin][i] += vect[i];
        }
    }

    public double[] executeKMeans(long[] metrics) {
//        long startTime = System.nanoTime();
        long counter = 0;
        do {
            F = S;
            kmean();
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

    public static void main(String[] args) throws Exception {
        long[] metrics = {0, 0, 0, 0, 0};
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
            double[] totalNormAvg = null;
            totalNormAvg = algorithm.executeKMeans(metrics);

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
//        System.out.println(Common.formatTable(G));
        Common.printList(G);
        long endTime = System.nanoTime();
        metrics[0] = endTime - startTime;
        Common.printMetrics(metrics, clustersNumber);
    }

    public static double norm(Double[] a, double[] b) {
        double res = 0;
        for (int i = 0; i < a.length; i++) {
            res = res + Math.pow(a[i] - b[i], 2);
        }
        return res;
    }
}