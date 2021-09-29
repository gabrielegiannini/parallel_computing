import java.util.*;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class MainWithThreads {

    public static void main(String[] args) throws Exception {
        long startTime = System.nanoTime();
        List<String> positionals = new LinkedList<String>();

        boolean initRandomClusters = Common.extractArguments(args, positionals);
        HashMap<String, Double[]> data = new HashMap<>();
        int n = Common.populateData(positionals, data);
        int clustersNumber = Integer.parseInt(positionals.get(1)); // args[0]

        final int execPerThread = data.size() / Common.THREAD_COUNT;
//        final int execSurplus = data.size() - execPerThread * Common.THREAD_COUNT;

        //crea i thread
        Thread[] threads = new Thread[Common.THREAD_COUNT];
        final CyclicBarrier barrierStart = new CyclicBarrier(Common.THREAD_COUNT + 1);
        final CyclicBarrier barrierStop = new CyclicBarrier(Common.THREAD_COUNT + 1);
        final List<Runnable>[] chunks = new List[Common.THREAD_COUNT];
        for (int i = 0; i < threads.length; i++) {
            chunks[i] = new ArrayList<>();
            final int finalI = i;
            threads[i] = new Thread(() -> {
                while (true) {
                    try {
                        barrierStart.await();
                        for (var task : chunks[finalI]) {
                            task.run();
                        }
                        barrierStop.await();
                    } catch (InterruptedException | BrokenBarrierException e) {
                        return;
                    }
                }
            });
            threads[i].start();
        }
        int c = 0;
        double minimunTotal = Double.MAX_VALUE;
        List<String>[] FINAL = null;
        long[] metrics = {0, 0, 0, 0, 0};
        while (c < threads.length) {
            final int finalC = c;

            KmeansPar algorithm = new KmeansPar(data, clustersNumber, n);
            if (initRandomClusters) {
                algorithm.initClusters();
            } else {
                algorithm.initMeans();
            }
            double[] totalNormAvg = null;
            totalNormAvg = algorithm.executeKMeans((task, id) -> {
                chunks[id % Common.THREAD_COUNT].add(task);
            }, () -> {
                try {
                    barrierStart.await();
                } catch (InterruptedException | BrokenBarrierException e) {
                    e.printStackTrace();
                }
            }, () -> {
                try {
                    barrierStop.await();
                } catch (InterruptedException | BrokenBarrierException e) {
                    e.printStackTrace();
                }
            }, metrics);

            double total = 0;
            for (double avg : totalNormAvg) {
                total = total + avg;
            }
            if (minimunTotal > total) {
                minimunTotal = total;
                FINAL = algorithm.getClusters();
            }

            for (int i = 0; i < Common.THREAD_COUNT; i++) {
                chunks[i].clear();
            }

            c++;
        }
        Common.printList(FINAL);
        long endTime = System.nanoTime();
        metrics[0] = endTime - startTime;
        Common.printMetrics(metrics, clustersNumber);
        for (var t : threads) {
            t.interrupt();
        }
    }

}
