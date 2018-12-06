import numpy as np
from skimage.transform import downscale_local_mean, resize
import os
from common.time_utils import get_local_datetime
from config.settings import MAP_WIDTH, MAP_HEIGHT, DATA_DIR, MIN_DISPATCH_CYCLE,\
    DESTINATION_PROFILE_SPATIAL_AGGREGATION, DESTINATION_PROFILE_TEMPORAL_AGGREGATION
from .settings import MAX_MOVE, NUM_SUPPLY_DEMAND_MAPS, FLAGS, AUX_LENGTH
from common import vehicle_status_codes, mesh
from .demand_loader import DemandLoader


L = MAX_MOVE * 2 + 1

class FeatureConstructor(object):

    def __init__(self):
        self.demand_loader = DemandLoader()
        self.t = 0
        self.fingerprint = (100000, 0)
        self.reachable_map = self.load_reachable_map()
        self.state_space = [(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT) if self.reachable_map[x, y] == 1]
        self.DT = self.load_dt_map()
        if FLAGS.average:
            self.D_out = self.D_in = np.ones((MAP_WIDTH, MAP_HEIGHT)) / (L ** 2)
        else:
            self.D_out, self.D_in = self.build_diffusion_filter()

        self.d_entropy = self.build_diffusion_entropy_map()
        self.TT = None
        self.OD = None
        self.x_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        self.y_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        self.d_matrix = np.zeros((AUX_LENGTH, AUX_LENGTH))
        for i in range(AUX_LENGTH):
            self.x_matrix[i, :] = i - AUX_LENGTH/2
            self.y_matrix[:, i] = i - AUX_LENGTH/2
            for j in range(AUX_LENGTH):
                self.d_matrix[i, j] = np.sqrt((i - AUX_LENGTH/2)**2 + (j - AUX_LENGTH/2)**2) / AUX_LENGTH
        self.legal_map = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.float32)
        for x in range(MAP_WIDTH):
            for y in range(MAP_HEIGHT):
                if self.is_reachable(x, y):
                    self.legal_map[x, y] = 1.0


    def action_space_iter(self, x, y):
        yield (0, 0)
        for ax in range(-MAX_MOVE, MAX_MOVE + 1):
            for ay in range(-MAX_MOVE, MAX_MOVE + 1):
                if ax == 0 and ay == 0:
                    continue
                x_ = x + ax
                y_ = y + ay
                if self.is_reachable(x_, y_):
                    yield (ax, ay)

    def load_reachable_map(self):
        return np.load(os.path.join(DATA_DIR, 'reachable_map.npy'))

    def load_dt_map(self):
        return np.load(os.path.join(DATA_DIR, 'tt_map.npy')) / MIN_DISPATCH_CYCLE

    # def build_diffusion_filter(self):
    #     D_out = np.exp(-(self.DT) ** 2 + 1) / (L ** 2)
    #     D_in = np.zeros((MAP_WIDTH, MAP_HEIGHT, L, L))
    #     for x, y in self.state_space:
    #         for ax, ay in self.action_space_iter(x, y):
    #             axi, ayi = MAX_MOVE + ax, MAX_MOVE + ay
    #             D_in[x, y, axi, ayi] = D_out[x + ax, y + ay, -axi-1, -ayi-1]
    #     return D_out, D_in
    #
    #
    # def build_diffusion_entropy_map(self):
    #     entropy = np.zeros((MAP_WIDTH, MAP_HEIGHT))
    #     for x, y in self.state_space:
    #         entropy[x, y] = -(self.D_out[x, y] * np.log(self.D_out[x, y] + 1e-6)).sum()
    #     entropy /= np.log(L ** 2 + 1e-6)
    #     diffused_entropy = [entropy] + self.diffusion_convolution(entropy, self.D_out, FLAGS.n_diffusions - 1)
    #     return diffused_entropy

    def update_time(self, current_time):
        self.t = current_time

    def update_supply(self, vehicles, duration=MIN_DISPATCH_CYCLE * 2):
        idle = vehicles[(vehicles.status == vehicle_status_codes.IDLE) | (vehicles.status == vehicle_status_codes.CRUISING)]
        occupied = vehicles[vehicles.status == vehicle_status_codes.OCCUPIED]
        occupied = occupied[occupied.time_to_destination <= duration]
        idle_map = self.construct_supply_map(idle[["lon", "lat"]].values)
        dropoff_map = self.construct_supply_map(occupied[["destination_lon", "destination_lat"]].values)
        self.supply_maps = [dropoff_map, idle_map]
        # self.diffused_supply = []
        # for s in self.supply_maps:
        #     self.diffused_supply += self.diffusion_convolution(s, self.D_in, FLAGS.n_diffusions)

    def update_demand(self, t, demand_normalized_factor=0.1, tt_normalized_factor=1.0/1800, horizon=1):
        profile, diff = self.demand_loader.load(t, horizon=horizon)
        self.demand_maps = [d * demand_normalized_factor for d in profile] # + [diff]
        # self.diffused_demand = []
        # for d in self.demand_maps:
        #     self.diffused_demand += self.diffusion_convolution(d, self.D_out, FLAGS.n_diffusions)

        # if FLAGS.trip_diffusion:
        #     if self.OD is None or t % (DESTINATION_PROFILE_TEMPORAL_AGGREGATION * 3600) == 0:
        #         self.OD, self.TT = self.demand_loader.load_OD_matrix(t)
        #         self.TT = resize(self.TT * tt_normalized_factor, (MAP_WIDTH, MAP_HEIGHT), mode='edge')
        #
        #     d = sum(self.diffused_demand[:horizon])
        #     self.diffused_demand.append(self.trip_diffusion_convolution(d, self.OD))
        #     self.diffused_demand.append(self.TT)

    # def diffusion_convolution(self, img, d_filter, k):
    #     M = img
    #     diffused_maps = []
    #     for _ in range(k):
    #         M = self.diffuse_map(M, d_filter)
    #         diffused_maps.append(M)
    #     return diffused_maps

    # def trip_diffusion_convolution(self, img, trip_filter):
    #     n = DESTINATION_PROFILE_SPATIAL_AGGREGATION
    #     M = downscale_local_mean(img, (n, n))
    #     M = np.tensordot(trip_filter, M)
    #     M = resize(M, img.shape, mode='edge')
    #     return M

    # def diffuse_map(self, img, d_filter):
    #     padded_map = np.pad(img, MAX_MOVE, "constant")
    #     diffused_map = self.construct_initial_map()
    #     for x, y in self.state_space:
    #         diffused_map[x, y] = (padded_map[x : x + L, y : y + L] * d_filter[x, y]).sum()
    #     return diffused_map

    def update_fingerprint(self, fingerprint):
        self.fingerprint = fingerprint

    def construct_current_features(self, x, y):
        # M = self.get_supply_demand_maps()
        # t = self.get_current_time()
        # f = self.get_current_fingerprint()
        # l = (x, y)
        # s, actions = self.construct_features(t, f, l, M)
        # return s, actions
        t = self.get_current_time()
        M = self.get_supply_demand_maps()
        main_maps = self.construct_main_maps(x, y, M) # 1 demand + 1 dropoff + 1 idle, each 51x51
        aux_maps = self.construct_aux_maps(x, y, t)   # 11 maps, each 15x15
        return main_maps + aux_maps

    def construct_main_maps(self, x, y, M):
        main_maps = []
        for m in M:
            main_maps.append(self.resize_map(m, x, y))
        return main_maps

    def resize_map(self, map, x, y, output_length=MAIN_LENGTH):
        m = np.pad(map, (output_length-1)/2, 'constant')
        return m[x:x+output_length, y:y+output_length]

    def construct_aux_maps(self, x, y, t):
        t = get_local_datetime(timestamp)
        dayofweek = t.weekday() / 7.0 * 2 * np.pi
        hourofday = t.hour / 24.0 * 2 * np.pi
        m_ones = np.ones((AUX_LENGTH, AUX_LENGTH), dtype=np.float32)

        m_dayofweek = [m_ones * np.sin(dayofweek), m_ones * np.cos(dayofweek)]
        m_hourofday = [m_ones * np.sin(hourofday), m_ones * np.cos(hourofday)]
        m_position = np.zeros((AUX_LENGTH, AUX_LENGTH), dtype=np.float32)
        m_position[AUX_LENGTH//2, AUX_LENGTH//2] = 1.0
        m_coordinate = [1.0*x/MAP_WIDTH * m_ones, 1.0*y/MAP_HEIGHT * m_ones]
        m_move_coordinate = [(float(x) + self.x_matrix) / MAP_WIDTH, (float(y) + self.y_matrix) / MAP_HEIGHT]
        m_distance = self.d_matrix
        m_sensible = self.resize_map(self.legal_map, x, y)
        return m_dayofweek + m_hourofday + m_position + m_coordinate + m_move_coordinate + m_distance + m_sensible

    def construct_features(self, t, f, l, M):
        # state_feature = self.construct_state_feature(t, f, l, M)
        # actions, action_features = self.construct_action_features(t, l, M)
        # s = (state_feature, action_features)
        # return s, actions
        x, y = l
        main_maps = self.construct_main_maps(x, y, M)
        aux_maps = self.construct_aux_maps(x, y, t)
        return main_maps + aux_maps


    # def construct_state_feature(self, t, f, l, M):
    #     x, y = l
    #     state_feature = [m.mean() for m in M[:NUM_SUPPLY_DEMAND_MAPS]]   # 5
    #     state_feature += [m[x, y] for m in M]                            # 20
    #     state_feature += [m[x, y] for m in self.d_entropy]               # 3
    #     state_feature += self.construct_time_features(t) + self.construct_fingerprint_features(f)   # 6
    #     return state_feature                                             # 34
    #
    # def construct_action_features(self, t, l, M):
    #     actions = []
    #     action_features = []
    #
    #     for ax, ay in self.action_space_iter(*l):                 # <= 2 * 7 + 1 = 15
    #         a = (ax, ay)
    #         feature = self.construct_action_feature(t, l, M, a)   # 24
    #         if feature is not None:
    #             actions.append(a)
    #             action_features.append(feature)
    #
    #     return actions, action_features

    def is_reachable(self, x, y):
        return 0 <= x and x < MAP_WIDTH and 0 <= y and y < MAP_HEIGHT and self.reachable_map[x, y] == 1

    # def construct_action_feature(self, t, l, M, a):
    #     x, y = l
    #     ax, ay = a
    #     x_ = x + ax
    #     y_ = y + ay
    #     tt = self.get_triptime(x, y, ax, ay)
    #     if tt <= 1:
    #         action_feature = [m[x_, y_] for m in M]                # 20
    #         action_feature += [m[x_, y_] for m in self.d_entropy]  # 3
    #         action_feature += [tt]                                 # 1
    #         # action_feature += self.construct_location_features((x_, y_)) + [tt]
    #         return action_feature
    #
    #     return None


    def get_triptime(self, x, y, ax, ay):
        return self.DT[x, y, ax + MAX_MOVE, ay + MAX_MOVE]

    def get_supply_demand_maps(self):
        # supply_demand_maps = self.supply_maps + self.demand_maps
        # diffused_maps = self.diffused_supply + self.diffused_demand
        # return supply_demand_maps + diffused_maps
        return self.demand_maps+self.supply_maps

    def construct_initial_map(self, w=MAP_WIDTH, h=MAP_HEIGHT):
        return np.zeros((w, h), dtype=np.float32)

    # def construct_supply_map(self, locations):
    #     supply_map = self.construct_initial_map()
    #     for lon, lat in locations:
    #         x, y = mesh.convert_lonlat_to_xy(lon, lat)
    #         supply_map[x, y] += 1.0
    #     return supply_map

    # def construct_time_features(self, timestamp):
    #     t = get_local_datetime(timestamp)
    #     hourofday = t.hour / 24.0 * 2 * np.pi
    #     dayofweek = t.weekday() / 7.0 * 2 * np.pi
    #     return [np.sin(hourofday), np.cos(hourofday), np.sin(dayofweek), np.cos(dayofweek)]

    # def construct_fingerprint_features(self, fingerprint):
    #     iteration, epsilon = fingerprint
    #     return [np.log(1 + iteration / 60.0), epsilon]

    # def construct_location_features(self, l):
    #     x, y = l
    #     x_norm = float(x - MAP_WIDTH / 2.0) / MAP_WIDTH * 2.0
    #     y_norm = float(y - MAP_HEIGHT / 2.0) / MAP_HEIGHT * 2.0
    #     return [x_norm, y_norm]

    def get_current_time(self):
        t = self.t
        return t

    def get_current_fingerprint(self):
        f = self.fingerprint
        return f