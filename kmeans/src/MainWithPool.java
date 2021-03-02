import java.io.IOException;
import java.util.*;

public class MainWithPool {

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        List<String> positionals = new LinkedList<String>();

        boolean initRandomClusters = Common.extractArguments(args, positionals);
        HashMap<String, Double[]> data = new HashMap<>();
        int n = Common.populateData(positionals, data);
        int clustersNumber = Integer.parseInt(positionals.get(1)); // args[0]

        final boolean finalInitRandomClusters = initRandomClusters;
        final int finalN = n;
        KMeanExecutor pool = new KMeanExecutor(Common.THREAD_COUNT);
        int c = 0;
        while (c < Common.EXECUTIONS_COUNT) {
            pool.schedule(() -> {
                Kmeans algorithm = null;
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
                algorithm.executeKMeans();
                return algorithm;
            });
            c++;
        }
        double globalAvg = Double.MAX_VALUE;
        List<String>[] FINAL = null;
        for (int i = 0; i < Common.EXECUTIONS_COUNT; i++) {
            double avg = 0;
            Kmeans task = pool.getNextTerminated();
            double[] normAvg = task.getTotalNormAvg();
            for (int j = 0; j < normAvg.length; j++) {
                avg = avg + normAvg[j];
            }
            if (avg < globalAvg) {
                globalAvg = avg;
                FINAL = task.getClusters();
            }
        }
        System.out.println(Common.formatTable(FINAL));
        pool.shutdown();
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in milliseconds : " + timeElapsed / 1000000);
    }
}
