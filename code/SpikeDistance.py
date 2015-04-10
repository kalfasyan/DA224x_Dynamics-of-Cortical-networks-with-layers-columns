import matplotlib.pyplot as plt
import pyspike as spk

spike_trains = spk.load_spike_trains_from_txt("PySpike_testdata.txt",
                                              time_interval=(0, 110))
spike_profile = spk.spike_profile(spike_trains[0], spike_trains[1])
x, y = spike_profile.get_plottable_data()
plt.plot(x, y, '--k')
print("SPIKE distance: %.8f" % spike_profile.avrg())
plt.show()

spike_dist = spk.spike_distance(spike_trains[0], spike_trains[1])
print spike_dist