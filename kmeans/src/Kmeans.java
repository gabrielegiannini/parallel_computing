import java.util.ArrayList;
import java.util.List;

public class Kmeans {
    public static void main(String[] args) throws Exception {
        double[] arrayInt = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        int clustersNumber = 3;
        double[] initialMeans = new double[clustersNumber];
        for (int i = 0; i < initialMeans.length; i++) {
            initialMeans[i] = Math.random() * 10;
        }
        List<Integer>[] S = kmeans(arrayInt, initialMeans);
        List<Integer>[] F;
        do {
            for (int i = 0; i < initialMeans.length; i++) {
                initialMeans[i] = 0;
                for (int numbers : S[i]) {
                    initialMeans[i] = initialMeans[i] + arrayInt[numbers];
                }
                initialMeans[i] = initialMeans[i] / S[i].size();
            }
            F = S;
            S = kmeans(arrayInt, initialMeans);
        } while (different(F, S));
        printList(S, arrayInt);
    }

    public static List<Integer>[] kmeans(double[] arrayInt, double[] initialMeans) {
        List<Integer>[] S = new ArrayList[3];
        for (int j = 0; j < S.length; j++) {
            S[j] = new ArrayList<>();
        }
        for (int k = 0; k < arrayInt.length; k++) {
            int posMin = 0;
            double min = 10;
            for (int h = 0; h < initialMeans.length; h++) {
                if (Math.abs(arrayInt[k] - initialMeans[h]) < min) {
                    min = Math.abs(arrayInt[k] - initialMeans[h]);
                    posMin = h;
                }
            }
            S[posMin].add(k);
        }
        return S;
    }

    public static boolean different(List<Integer>[] F, List<Integer>[] S) {
        boolean no = false;
        for (int i = 0; i < F.length; i++) {
            if (!F[i].equals(S[i])) {
                no = true;
                break;
            }
        }
        return no;
    }

    public static void printList(List<Integer>[] S, double[] arrayInt) {
        for (int f = 0; f < S.length; f++) {
            for (int number : S[f]) {
                System.out.print(arrayInt[number] + ", ");
            }
            System.out.println(".");
        }
    }
}