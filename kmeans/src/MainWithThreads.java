import java.io.IOException;
import java.util.*;

public class MainWithThreads {

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        List<String> positionals = new LinkedList<String>();

        boolean initRandomClusters = Common.extractArguments(args, positionals);
        HashMap<String, Double[]> data = new HashMap<>();
        int n = Common.populateData(positionals, data);
        int clustersNumber = Integer.parseInt(positionals.get(1)); // args[0]

        double[][] totalNormAvgR = new double[Common.THREAD_COUNT][];
        List<String>[][] G = new List[Common.THREAD_COUNT][];

        final boolean finalInitRandomClusters = initRandomClusters;
        final int finalN = n;
        final int execPerThread = Common.EXECUTIONS_COUNT / Common.THREAD_COUNT;
        final int execSurplus = Common.EXECUTIONS_COUNT - execPerThread * Common.THREAD_COUNT;
        Thread[] threads = new Thread[Common.THREAD_COUNT];
        int c = 0;
        long[] metrics = {0, 0, 0, 0, 0};
        while (c < threads.length) {
            final int finalC = c;
            Thread t = new Thread(() -> {
                List<String>[][] F = new List[execPerThread + 1][];
                double[][] totalNormAvg = new double[execPerThread + 1][];
                int a = 0;
                Kmeans algorithm = null;
                double globalAvg = Double.MAX_VALUE;
                int globalAvgIndex = 0;
                while (a < execPerThread || (a < execPerThread + 1 && finalC < execSurplus)) {
                    algorithm = null;
                    try {
                        algorithm = new Kmeans(data, clustersNumber, finalN);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    if (finalInitRandomClusters) {
                        algorithm.initClusters();
                    } else {
                        algorithm.initMeans();
                    }
                    totalNormAvg[a] = algorithm.executeKMeans(metrics);
                    F[a] = algorithm.getClusters();
                    a++;
                }

                for (int i = 0; i < totalNormAvg.length; i++) {
                    if (i >= execPerThread && finalC >= execSurplus)
                        break;
                    double avg = 0;
                    for (int j = 0; j < totalNormAvg[i].length; j++) {
                        avg = avg + totalNormAvg[i][j];
                    }
                    if (avg < globalAvg) {
                        globalAvg = avg;
                        globalAvgIndex = i;
                    }
                }
                totalNormAvgR[finalC] = totalNormAvg[globalAvgIndex];
                G[finalC] = F[globalAvgIndex];// prende il nome del thread e ci mette il
                // nuovo cluster
            });
            t.start();
            threads[c] = t;
            c++;
        }
        // poi qui si confrontano tutti i valori di G e si prende il migliore.
        double globalAvg = Double.MAX_VALUE;
        int globalAvgIndex = 0;
        for (int i = 0; i < totalNormAvgR.length; i++) {
            double avg = 0;
            threads[i].join(); // aspetta che abbia finito
            for (int j = 0; j < totalNormAvgR[i].length; j++) {
                avg = avg + totalNormAvgR[i][j];
            }
            if (avg < globalAvg) {
                globalAvg = avg;
                globalAvgIndex = i;
            }
        }
//        System.out.println(Common.formatTable(G[globalAvgIndex]));
        Common.printList(G[globalAvgIndex]);
        long endTime = System.nanoTime();
        metrics[0] = endTime - startTime;
        Common.printMetrics(metrics, clustersNumber);
    }

}
