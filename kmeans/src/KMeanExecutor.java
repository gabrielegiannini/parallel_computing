import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.*;

public class KMeanExecutor {
    private ExecutorService executor;
    private List<Future<Kmeans>> results = new ArrayList<>();

    public KMeanExecutor(int threadCount) {
        executor = Executors.newFixedThreadPool(threadCount);
    }

    public void schedule(Callable<Kmeans> callable) {
        results.add(executor.submit(callable));
    }

    public Kmeans getNextTerminated() {
        Optional<Future<Kmeans>> result = null;
        while (results.parallelStream().noneMatch(f -> f.isDone())) {
            try {
                Thread.sleep(2);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        result = results.parallelStream().filter(f -> f.isDone()).findAny();
        if (result.isPresent()) {
            try {
                results.remove(result);
                return result.get().get();
            } catch (InterruptedException e) {
                e.printStackTrace();
            } catch (ExecutionException e) {
                e.printStackTrace();
            }
        } else {
            return null;
        }
        return null;
    }

    public void shutdown() {
        executor.shutdown();
    }
}
