import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainWithPool {

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        int c = 0;
        List<String> positionals = new LinkedList<String>();
        boolean initRandomClusters = false;
        while (c < args.length) {
            if ("-initClusters".equals(args[c])) {
                initRandomClusters = true;
            } else {
                positionals.add(args[c]);
            }
            c++;
        }
        BufferedReader csv = null;
        try {
            csv = new BufferedReader(new FileReader(positionals.get(0)));
        } catch (FileNotFoundException e) {
            System.err.println("given path does not exists");
            System.exit(1);
        }

        String row = csv.readLine();
        HashMap<String, Double[]> data = new HashMap<>();
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

        int clustersNumber = Integer.parseInt(positionals.get(1)); // args[0]
        Random r = new Random();
        c = 0;

        final int executionsCount = 100000;
        final int threadsCount = 16;
        final boolean finalInitRandomClusters = initRandomClusters;
        final int finalN = n;
        Thread[] threads = new Thread[100];
        KMeanExecutor pool = new KMeanExecutor(threadsCount);
        while (c < executionsCount) {
            final int finalC = c;
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
        for (int i = 0; i < executionsCount; i++) {
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
        System.out.println(Main.formatTable(FINAL));
        pool.shutdown();
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in milliseconds : " + timeElapsed / 1000000);
    }
}
