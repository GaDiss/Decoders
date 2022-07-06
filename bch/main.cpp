#pragma GCC optimize("O2")

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <set>

#include <cstdio>

using namespace std;

template<typename T>
inline vector<T> read_vector(int n) {
    vector<T> v(n);
    for (int j = 0; j < n; j++) {
        T b;
        cin >> b;
        v[j] = b;
    }
    return v;
}

inline vector<bool> mul_poly(vector<bool> &a, vector<bool> &b) {
    vector<bool> r(a.size() + b.size() - 1, false);
    for (int i = 0; i < b.size(); i++) {
        for (int j = 0; j < a.size(); j++) {
            r[i + j] = r[i + j] xor (b[i] && a[j]);
        }
    }
    return r;
}

class BCH {
public:
    int n, d, r, k;
    vector<int> F, Fr;     // Field table
    vector<vector<int>> M; // multiplicity table
    vector<bool> g;        // generating polynomial

    BCH(int n, int a, int d) {
        this->n = n;
        this->d = d;

        // highest bit of a (supposedly n + 1)
        int aa = a;
        int hb = 1;
        while (aa >>= 1) hb <<= 1;

        // Field table
        F.push_back(1);
        Fr = vector<int>(n + 1, -1);
        Fr[1] = 0;
        for (int i = 1; i < n; i++) {
            F.push_back(F[i - 1] << 1);
            if (F[i] >= hb) F[i] ^= a;
            Fr[F[i]] = i;
        }

        // generate multiplicity table
        M = vector<vector<int>>(n + 1, vector<int>(n + 1, 0));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                M[i][j] = F[(Fr[i] + Fr[j]) % n];
            }
        }

        // Find cyclotomic classes
        set<int> all_c;
        for (int i = 1; i < n; i++) all_c.insert(i);
        vector<set<int>> cyclotomic_classes;
        int ci = 0;
        while (!all_c.empty()) {
            int c = *all_c.begin();
            if (c >= d) break;
            set<int> cs;
            while (cs.find(c) == cs.end()) {
                cs.insert(c);
                all_c.erase(c);
                c = (2 * c) % n;
            }
            cyclotomic_classes.push_back(cs);
            ci++;
        }

        // calc generating polynomial
        g = {true};
        for (const auto &cl : cyclotomic_classes) {
            // minimal polynomial of cyclotomic class
            vector<int> poly = {0};
            for (auto root : cl) {
                // poly1(x) = poly(x) * x
                vector<int> poly1 = {-1};
                for (auto p : poly) poly1.push_back(p);

                // poly(x) = poly1 + poly(x) * root
                for (int i = 0; i < poly.size(); i++) {
                    if (poly[i] >= 0) poly[i] = (poly[i] + root) % n;
                    if (poly[i] >= 0 && poly1[i] >= 0) {
                        poly[i] = Fr[F[poly[i]] ^ F[poly1[i]]];
                    } else if (poly1[i] >= 0) {
                        poly[i] = poly1[i];
                    }
                }
                poly.push_back(poly1[poly1.size() - 1]);
            }
            // transform poly to vector<bool>
            vector<bool> b;
            transform(poly.cbegin(), poly.cend(), back_inserter(b),
                      [](int bb) -> bool { return bb == 0; }
            );
            g = mul_poly(g, b);
        }

        r = (int) g.size() - 1;
        k = n - r;

        cout << k << '\n';
        for (auto b : g) {
            cout << b << ' ';
        }
        cout << '\n';
    }

    vector<bool> encode(vector<bool> &v) {
        // res = v * x^r + remainder
        vector<bool> res(r, false);
        res.insert(res.end(), v.cbegin(), v.cend());

        // calc remainder after dividing res by g
        vector<bool> temp = res;
        for(int i = res.size() - 1; i >= r; i--) {
            if (temp[i]) {
                for (int j = 0; j < g.size(); j++) {
                    temp[i - j] = temp[i - j] xor g[r - j];
                }
            }
        }
        // add reminder to res
        for(int i = 0; i < r; i++) res[i] = temp[i];

        return res;
    }

    void decode(vector<bool> &v) {

        // find syndrome
        vector<int> s;
        int count_zero_s = 0;
        for (int j = 1; j <= d - 1; j++) {
            int t = 0;
            for (int i = 0; i < v.size(); i++) {
                if (v[i]) t ^= F[(i * j) % n];
            }
            if (t == 0) count_zero_s++;
            s.push_back(t);
        }

        // if all syndromes are zero, there are no errors
        if (count_zero_s == d - 1) return;

        // Berlekamp–Massey algorithm
        int L = 0; // current register length
        int m = 0;
        vector<int> lambda = {1}; // locators polynomial
        vector<int> b = {1}; // discrepancy compensating polynomial
        for (int rr = 1; rr <= d - 1; rr++) {
            int delta = 0; // discrepancy
            for (int j = 0; j <= min(L, (int)lambda.size()); j++) {
                delta ^= M[lambda[j]][s[rr - j - 1]];
            }

            if (delta != 0) {
                vector<int> temp = lambda;
                temp.resize(max(temp.size(), rr - m + b.size()), 0);
                for (int i = 0; i < b.size(); i++) temp[rr - m + i] ^= M[delta][b[i]];

                if (2 * L <= rr - 1) {
                    b.clear();
                    int dd = F[(n - Fr[delta]) % n];
                    for (int i : lambda) {
                        b.push_back(M[dd][i]);
                    }
                    L = rr - L;
                    m = rr;
                }

                lambda = temp;
            }
        }

        // if L != deg(lambda), there are too many errors
        if (L != lambda.size() - 1) return;

        // locate error positions
        vector<int> err;
        // if lambda(a^-x) = 0, there is an error in position x
        for (int i = 0; i < n; i++) {
            int res = lambda[0];
            for (int j = 1; j < lambda.size(); j++) {
                res ^= M[F[(j * i) % n]][lambda[j]];
            }
            if (res == 0) {
                err.push_back((n - i) % n);
                if (err.size() == L) break;
            }
        }

        // correct all errors
        for (int e : err) v[e] = !v[e];
    }
};

void run() {
    int n, a, d;
    cin >> n >> a >> d;

    auto coder = BCH(n, a, d);
    int k = coder.k;

    // for random in Simulate
    std::random_device rd{};
    std::mt19937 gen{rd()};
    auto gen_b = bind(uniform_int_distribution<>(0, 1), default_random_engine());
    uniform_real_distribution<> dist(0, 1);

    string command;
    while (cin >> command) {
        if (command == "Encode") {
            vector<bool> b = read_vector<bool>(k);
            b = coder.encode(b);
            for (bool i : b) cout << i << " ";
            cout << '\n';
        }

        if (command == "Decode") {
            auto b = read_vector<bool>(n);
            coder.decode(b);
            for (bool i : b) cout << i << " ";
            cout << '\n';
        }

        if (command == "Simulate") {
            double p;
            int num_of_simulations, max_error;
            cin >> p >> num_of_simulations >> max_error;

            int errs = 0;
            int iters = 0;
            for (int i = 0; i < num_of_simulations; i++) {
                vector<bool> b = vector<bool>(k);
                for (int j = 0; j < k; j++) b[j] = gen_b();
                b = coder.encode(b);
                vector<bool> b2;
                transform(b.cbegin(), b.cend(), back_inserter(b2),
                          [&dist, &gen, &p](bool bb) -> bool {
                              return (dist(gen) <= p) ? !bb : bb;
                          });
                coder.decode(b2);
                if (b != b2) errs++;
                iters = i + 1;
                if (errs >= max_error) break;
            }
            cout << (double) errs / iters << '\n';
        }
    }
}

int main() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    cin.tie(nullptr);
    cout.tie(nullptr);

    run();
}
