import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class KmeansND {

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

    public KmeansND(final Map<String, Double[]> data, int clustersNumber, final int n) throws IOException {
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
        int cont = 0;
        for (Entry<String, Double[]> entry : data.entrySet()) {
            int posMin = 0;

            double min = Double.MAX_VALUE;
            for (int h = 0; h < means.length; h++) {
                double norm = norm(entry.getValue(), means[h]);
                cont++;
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
        // return S;
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
        List<String>[] G = null;
        while (c < 100) {
            KmeansND algorithm = new KmeansND(data, clustersNumber, n);
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
        System.out.println(formatTable(G));
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

    public static List<String>[] kmeans(Map<String, Double[]> data, double[][] initialMeans, double[] totalNormAvg) {
        List<String>[] S = new ArrayList[initialMeans.length];// dimensione clusterNumber
        for (int j = 0; j < S.length; j++) {
            S[j] = new ArrayList<>();
        }
        for (int h = 0; h < totalNormAvg.length; h++) {
            totalNormAvg[h] = 0;
        }
        int cont = 0;
        for (Entry<String, Double[]> entry : data.entrySet()) {
            int posMin = 0;

            double min = Double.MAX_VALUE;
            for (int h = 0; h < initialMeans.length; h++) {
                double norm = norm(entry.getValue(), initialMeans[h]);
                cont++;
                if (norm < min) {
                    min = norm;
                    posMin = h;
                }
            }
            S[posMin].add(entry.getKey());
            totalNormAvg[posMin] = totalNormAvg[posMin] + min;
        }
        // System.out.println("norm calcolata " + cont + " volte");
        for (int i = 0; i < totalNormAvg.length; i++) {
            if (S[i].size() > 0) {
                totalNormAvg[i] = totalNormAvg[i] / S[i].size();
            }
        }
        return S;
    }

    public static boolean different(List<String>[] F, List<String>[] S) {
        boolean no = false;
        for (int i = 0; i < F.length; i++) {
            if (!F[i].equals(S[i])) {
                no = true;
                break;
            }
        }
        return no;
    }

    public static void printList(Iterable<String>[] S) {
        for (Iterable<String> strings : S) {
            for (String key : strings) {
                System.out.print(key + ", ");
            }
            System.out.println(".");
        }
    }

    public static void printAnyThing(Object S) {
        if (S instanceof Object[]) {
            Object[] oArr = (Object[]) S;
            for (Object o : oArr) {
                printAnyThing(o);
                System.out.print(",");
            }
            System.out.println(".");
            System.out.println();
        } else if (S instanceof Iterable) {
            Iterable<Object> oIter = (Iterable<Object>) S;
            for (Object o : oIter) {
                printAnyThing(o);
                System.out.print(",");
            }
            System.out.println(".");
            System.out.println();
        } else {
            System.out.print(S.toString());
        }
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

    public static double norm(Double[] a, double[] b) {
        double res = 0;
        for (int i = 0; i < a.length; i++) {
            res = res + Math.pow(a[i] - b[i], 2);
        }
        return res;
    }

    public static double[][] mean(double[][] initialMeans, List<String>[] S, Map<String, Double[]> data) {
        for (int i = 0; i < initialMeans.length; i++) {
            for (int j = 0; j < initialMeans[i].length; j++) {
                initialMeans[i][j] = 0;
                for (String key : S[i]) {
                    initialMeans[i][j] = initialMeans[i][j] + data.get(key)[j];
                }
                initialMeans[i][j] = initialMeans[i][j] / S[i].size();
            }
        }
        return initialMeans;
    }
}