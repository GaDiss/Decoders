#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <functional>
#include <random>
#include <set>
#include <map>

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

inline bool mul(vector<bool>& a, vector<bool>& b) {
    vector<bool> r;
    transform(a.cbegin(), a.cend(), b.cbegin(), back_inserter(r), logical_and<>());
    return accumulate(r.begin(), r.end(), false, bit_xor<>());
}

inline void add(vector<bool>& a, vector<bool>& b) {
    transform(a.cbegin(), a.cend(), b.cbegin(), a.begin(), bit_xor<>());
}

class Viterbi {
private:
    int n;
    vector<vector<bool>> G;  // generating matrix
    vector<vector<bool>> Gt; // transposed G

    vector<vector<pair<bool, int>>> from; // grid edges
    vector<int> layers;                   // grid layers

public:
    Viterbi(int n, int k) {
        this->n = n;

        // read generating matrix
        Gt = vector<vector<bool>>(n);
        for (int i = 0; i < k; i++) {
            G.push_back(read_vector<bool>(n));
            for (int j = 0; j < n; j++) {
                Gt[j].push_back(G[G.size() - 1][j]);
            }
        }

        // transform to minimal spanning form
        // make span starts unique with gaussian elimination
        for (int i = 0, col = 0; i < k && col < n; i++, col++) {
            int j = k;
            while(j >= k && col < n) {
                j = i;
                while (j < k && !G[j][col]) j++;
                if (j >= k) col++;
            }
            if (col >= n) break;
            if (i != j) add(G[i], G[j]);
            for (j = i + 1; j < k; j++) {
                if (G[j][col]) add(G[j], G[i]);
            }
        }
        // make span ends unique
        set<int> rows; // rows with not determined unique span end
        for (int i = 0; i < k; i++) rows.insert(i);
        for (int col = n - 1; col >= 0; col--) {
            int m = -1;
            int count = 0;
            for (auto r : rows) {
                if (G[r][col]) {
                    count++;
                    m = max(r, m);
                }
            }
            if (m == -1) continue;
            rows.erase(m);
            if (count == 1) continue;
            for (int j = m - 1; j >= 0; j--) {
                if (G[j][col]) add(G[j], G[m]);
            }
        }

        // calculate spans
        vector<int> news(n, -1); // news[i] = j if j's span start = i
        vector<int> olds(n, -1); // olds[i] = j if j's span end = i - 1
        for (int i = 0; i < k; i++) {
            int l = 0;
            int r = n - 1;
            while (!G[i][l]) l++;
            while (!G[i][r]) r--;
            news[l] = i;
            olds[r] = i;
        }

        // build grid
        from.emplace_back(); // start node
        layers.push_back(0); // first layer
        map<vector<bool>, int> prev_layer = {{{}, 0}}; // remember labels of (i-1)th layer
        set<int> active; // currently active rows
        for (int i = 0; i < n; i++) {
            map<vector<bool>, int> nodes; // nodes on (i)th layer

            vector<bool> cur_col; // active bits of (i)th column of G
            cur_col.reserve(active.size());
            for (int a : active) cur_col.push_back(G[a][i]);
            if (news[i] != -1) cur_col.push_back(G[news[i]][i]);

            // generate nodes and edges on (i)th layer
            for (const auto& p : prev_layer) {
                vector<bool> l_id = p.first;
                vector<bool> new_l_id, new_l_id2; // labels for new nodes
                int j = -1;
                for (int a : active) {
                    j++;
                    if (olds[i] == a) continue;
                    new_l_id.push_back(l_id[j]);
                }
                if (news[i] != -1) {
                    new_l_id2 = new_l_id;
                    new_l_id.push_back(false);
                    new_l_id2.push_back(true);
                    l_id.push_back(false);
                }

                bool e_v = mul(l_id, cur_col); // edge value
                if (nodes.find(new_l_id) != nodes.end()) {
                    from[nodes[new_l_id]].push_back({e_v, p.second});
                    if (news[i] != -1) {
                        from[nodes[new_l_id2]].push_back({!e_v, p.second});
                    }
                } else {
                    from.push_back({{e_v, p.second}});
                    nodes.insert({new_l_id, from.size() - 1});
                    if (news[i] != -1) {
                        from.push_back({{!e_v, p.second}});
                        nodes.insert({new_l_id2, from.size() - 1});
                    }
                }
            }

            prev_layer = nodes;

            if (news[i] != -1) active.insert(news[i]);
            if (olds[i] != -1) active.erase(olds[i]);

            layers.push_back(from.size() - 1);
        }
    }

    vector<bool> encode(vector<bool> &v) {
        vector<bool> a;
        a.reserve(n);
        // a = v * G
        for (int i = 0; i < n; i++) {
            a.push_back(mul(v, Gt[i]));
        }
        return a;
    }

    vector<bool> decode(vector<double> &v) {
        /**
         * for every node calculate:
         *      - min difference between v and path from start
         *      - previous node on this path
         *      - value of last edge on this path
         */
        vector<tuple<double, int, bool>> d = {{0.0, -1, true}};

        for (int l = 0; l < n; l++) { // l is the current layer on the grid
            for (int i = layers[l] + 1; i <= layers[l + 1]; i++) { // i is the current node
                int f = -1;
                double min_err = std::numeric_limits<double>::infinity();
                bool t = true;
                for (auto p : from[i]) {
                    double e = get<0>(d[p.second]) + abs(v[l] - (p.first ? -1 : 1));
                    if (e < min_err) {
                        min_err = e;
                        f = p.second;
                        t = p.first;
                    }
                }
                d.emplace_back(min_err, f, t);
            }
        }

        // trace back the path with min difference from v
        auto cur = d.size() - 1;
        vector<bool> a(n);
        for (int i = n - 1; i >= 0; i--) {
            a[i] = get<2>(d[cur]);
            cur = get<1>(d[cur]);
        }
        return a;
    }

    void printL() {
        cout << 1 << ' ';
        for (int i = 1; i < layers.size(); i++) {
            cout << layers[i] - layers[i - 1] << ' ';
        }
        cout << endl;
    }
};

int main() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    cin.tie(nullptr);
    cout.tie(nullptr);

    int n, k;
    cin >> n >> k;

    Viterbi coder = Viterbi(n, k);
    coder.printL();

    // random for Simulate
    std::random_device rd{};
    std::mt19937 gen{rd()};
    auto gen_b = bind(uniform_int_distribution<>(0, 1), default_random_engine());

    string command;
    while (cin >> command) {
        if (command == "Encode") {
            vector<bool> b = read_vector<bool>(k);
            b = coder.encode(b);
            for (bool i : b) cout << i << " ";
            cout << endl;
        }

        if (command == "Decode") {
            vector<double> d = read_vector<double>(n);
            vector<bool> b = coder.decode(d);
            for (bool i : b) cout << i << " ";
            cout << endl;
        }

        if (command == "Simulate") {
            double snrb;
            int num_of_simulations, max_error;
            cin >> snrb >> num_of_simulations >> max_error;

            // calculate sigma for normal distribution
            double snr = pow(10, snrb / 10);
            snr = snr * k / n;
            double sigma = sqrt(1.0 / 2.0 / snr);
            normal_distribution<> nd{0, sigma};

            int errs = 0;
            int iters = 0;
            for (int i = 0; i < num_of_simulations; i++) {
                vector<bool> b = vector<bool>(k);
                for (int j = 0; j < k; j++) b[j] = gen_b();
                b = coder.encode(b);
                // add noise
                vector<double> d;
                transform(b.begin(), b.end(), back_inserter(d),
                          [&nd, &gen](bool bb) -> double {
                              return 1 - 2 * (bb ? 1 : 0) + nd(gen);
                          });
                vector<bool> b2 = coder.decode(d);
                if (b != b2) errs++;
                iters = i + 1;
                if (errs >= max_error) break;
            }
            cout << (double) errs / iters << endl;
        }
    }
}
