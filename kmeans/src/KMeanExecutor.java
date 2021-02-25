import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.*;

public class KMeanExecutor {
    private ExecutorService executor;
    private List<Future<KmeansND>> results = new ArrayList<>();
//    private final static FutureTask<KmeansND> empty = new FutureTask(() -> {
//    }, null); // placeholder

    public KMeanExecutor(int threadCount) {
        executor = Executors.newFixedThreadPool(threadCount);
//        empty.run();
    }

    public void schedule(Callable<KmeansND> callable) {
        results.add(executor.submit(callable));
    }

    public KmeansND getNextTerminated() {
        Optional<Future<KmeansND>> result = null;
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
