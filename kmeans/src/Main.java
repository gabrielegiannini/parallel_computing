import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Main {

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
        double minimunTotal = Double.MAX_VALUE;
        // List<double[]> totalNormAvg = new ArrayList<>();
        // List<List<String>[]> G = new ArrayList<>();// da controllare le variabili
        // comuni a tutti i thread se ci si
        // accede in modo
        // giusto, soprattuto G.

        // ptovo con degli array invece che con liste perché almeno sono già lunghi 100
        // posizioni

        double[][] totalNormAvg = new double[100][];
        List<String>[][] G = new List[100][];

        final boolean finalInitRandomClusters = initRandomClusters;
        final int finalN = n;
        Thread[] threads = new Thread[100];
        while (c < threads.length) {
            final int finalC = c;
            Thread t = new Thread(() -> {
                KmeansND algorithm = null;
                try {
                    algorithm = new KmeansND(data, clustersNumber, finalN);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                if (finalInitRandomClusters) {
                    algorithm.initClusters();
                } else {
                    algorithm.initMeans();
                }
                totalNormAvg[finalC] = algorithm.executeKMeans();

                /*
                 * double total = 0; for (double avg : totalNormAvg) { total = total + avg; }
                 */

                G[finalC] = algorithm.getClusters();// prende il nome del thread e ci mette il
                                                    // nuovo cluster
            });
            // t.setName(String.valueOf(c));
            // t.run(initRandomClusters, totalNormAvg, minimunTotal, G);
            t.start();
            threads[c] = t;
            // System.out.println(t.getName());
            c++;
        }
        // poi qui si confrontano tutti i valori di G e si prende il migliore.
        double globalAvg = Double.MAX_VALUE;
        int globalAvgIndex = 0;
        for (int i = 0; i < totalNormAvg.length; i++) {
            double avg = 0;
            threads[i].join(); // aspetta che abbia finito
            for (int j = 0; j < totalNormAvg[i].length; j++) {
                avg = avg + totalNormAvg[i][j];
            }
            if (avg < globalAvg) {
                globalAvg = avg;
                globalAvgIndex = i;
            }
        }
        System.out.println(formatTable(G[globalAvgIndex]));
        long endTime = System.nanoTime();
        long timeElapsed = endTime - startTime;
        System.out.println("Execution time in milliseconds : " + timeElapsed / 1000000);
        // formatTable ogni tanto potrebbe scazzare l'impaginazione (soprattutto per
        // datasetProva.csv cha ha nomi di
        // una sola lettera) in caso meglio usare printList (però con lui non si vedono
        // bene i cluster di test_reale.csv)
        // comunque la cosa che ci faceva vedere tutti i cluster in ordine era un errore
        // in format table, che ho più
        // o meno risotto
        // printList(G);
    }

    public static String formatTable(Object o) {
        StringBuilder sb = new StringBuilder();
        Formatter form = new Formatter(sb);
        if (o instanceof Object[]) {
            Object[] arr = (Object[]) o;
            List<List<String>> content = new ArrayList<>(arr.length);
            for (Object e : arr) {
                content.add(mapsObj(e));
            }
            for (int i = 0; i < arr.length; i++) {
                form.format("\t%d", i);
            }
            sb.append("\n");
            int j = 0;
            boolean next = true;
            String last = null;
            while (next) {
                next = false;
                for (int i = 0; i < arr.length; i++) {
                    List<String> col = content.get(i);
                    if (j < col.size()) {
                        String elem = col.get(j);
                        last = elem;
                        form.format("\t%s", elem);
                        if (j + 1 < col.size()) {
                            next = true;
                        }
                    } else {
                        if (last != null) { // questo aiuta
                            // ma questa funzione è ancora buggata...pace serve solo a stampare
                            int offset = last.length() / 8 + 1;
                            for (int h = 0; h < offset; h++) {
                                form.format("\t");
                            }
                        } else {
                            form.format("\t\t");
                        }
                    }
                }
                sb.append("\n");
                j++;
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    public static List<String> mapsObj(Object o) {
        List<String> lt = new LinkedList<>();
        if (o instanceof Iterable) {
            Iterable<Object> it = (Iterable<Object>) o;
            for (Object e : it) {
                lt.add(e.toString());
            }
        }
        return lt;
    }

}
