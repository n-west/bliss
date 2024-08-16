
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <core/hit.hpp>
#include <file_types/cpnp_files.hpp>

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdio> // std::tmpfile

TEST_CASE("capnp hits", "[serialization]") {
    SECTION("list of hits", "round trip a list of hits") {

        std::list<bliss::hit> original_hits;
        original_hits.push_back({
            .start_freq_index = 10,
            .start_freq_MHz = 1000.0F,
            .start_time_sec = 4.2,
            .duration_sec = 17,
            .rate_index = 5,
            .drift_rate_Hz_per_sec = 2.1,
            .power = 10000,
            .time_span_steps = 16,
            .snr = 10.0,
            .bandwidth = 3,
            .binwidth = 1
        });

        auto tempfname = std::tmpnam(nullptr);

        bliss::write_hits_to_capnp_file(original_hits, tempfname);

        auto deserialized_hits = bliss::read_hits_from_capnp_file(tempfname);

        REQUIRE(deserialized_hits.size() == original_hits.size());

        auto deserialized_hit_iter = deserialized_hits.begin();
        auto origianl_hit_iter = original_hits.begin();
        do {
            REQUIRE(deserialized_hit_iter->start_freq_index == origianl_hit_iter->start_freq_index);
            REQUIRE(deserialized_hit_iter->start_freq_MHz == Catch::Approx(origianl_hit_iter->start_freq_MHz));
            REQUIRE(deserialized_hit_iter->start_time_sec == Catch::Approx(origianl_hit_iter->start_time_sec));
            REQUIRE(deserialized_hit_iter->duration_sec == Catch::Approx(origianl_hit_iter->duration_sec));
            REQUIRE(deserialized_hit_iter->rate_index == origianl_hit_iter->rate_index);
            REQUIRE(deserialized_hit_iter->drift_rate_Hz_per_sec == Catch::Approx(origianl_hit_iter->drift_rate_Hz_per_sec));
            REQUIRE(deserialized_hit_iter->power == Catch::Approx(origianl_hit_iter->power));
            REQUIRE(deserialized_hit_iter->time_span_steps == origianl_hit_iter->time_span_steps);
            REQUIRE(deserialized_hit_iter->snr == Catch::Approx(origianl_hit_iter->snr));
            REQUIRE(deserialized_hit_iter->bandwidth == Catch::Approx(origianl_hit_iter->bandwidth));
            REQUIRE(deserialized_hit_iter->binwidth == origianl_hit_iter->binwidth);

        } while(++deserialized_hit_iter != deserialized_hits.end() && ++origianl_hit_iter != original_hits.end());
    }
}

TEST_CASE("capnp coarse_channel hits", "[serialization]") {
    SECTION("coarse_channel with hits", "round trip a coarse_channel that has hits") {

        auto coarse_channels = std::map<int, std::shared_ptr<bliss::coarse_channel>>();
        auto original_cc = bliss::coarse_channel(1000 /* fch1 */,
                                                1 /* foff */,
                                                0 /* machine_id */,
                                                32 /* nbits */,
                                                1000000 /* nchans */,
                                                18 /* nsteps */,
                                                1 /* nifs */,
                                                "test" /* source_name */,
                                                4 /* src_dej */,
                                                2 /* src_raj */,
                                                6 /* telescope_id */,
                                                17 /* tsamp */,
                                                54321 /* tstart */,
                                                1 /* datatype */,
                                                0 /* az_start */,
                                                0 /* za_start */);
        std::list<bliss::hit> original_hits;
        original_hits.push_back({
            .start_freq_index = 10,
            .start_freq_MHz = 1010.0F,
            .start_time_sec = 4.2,
            .duration_sec = 17,
            .rate_index = 5,
            .drift_rate_Hz_per_sec = 2.1,
            .power = 10000,
            .time_span_steps = 16,
            .snr = 10.0,
            .bandwidth = 3,
            .binwidth = 1
        });
        original_cc.add_hits(original_hits);
        
        auto tempfname = std::tmpnam(nullptr);

        bliss::write_coarse_channel_hits_to_capnp_file(original_cc, tempfname);

        // Targets serialized to disk -- Read back and check round-trip

        auto deserialized_cc = bliss::read_coarse_channel_hits_from_capnp_file(tempfname);

        REQUIRE(deserialized_cc.fch1() == original_cc.fch1());
        REQUIRE(deserialized_cc.foff() == original_cc.foff());
        REQUIRE(deserialized_cc.nchans() == original_cc.nchans());
        REQUIRE(deserialized_cc.tstart() == original_cc.tstart());
        REQUIRE(deserialized_cc.tsamp() == original_cc.tsamp());
        REQUIRE(deserialized_cc.ntsteps() == original_cc.ntsteps());
        REQUIRE(deserialized_cc.source_name() == original_cc.source_name());

        auto deserialized_hits = deserialized_cc.hits();

        REQUIRE(deserialized_hits.size() == original_hits.size());

        auto deserialized_hit_iter = deserialized_hits.begin();
        auto origianl_hit_iter = original_hits.begin();
        do {
            REQUIRE(deserialized_hit_iter->start_freq_index == origianl_hit_iter->start_freq_index);
            REQUIRE(deserialized_hit_iter->start_freq_MHz == Catch::Approx(origianl_hit_iter->start_freq_MHz));
            REQUIRE(deserialized_hit_iter->start_time_sec == Catch::Approx(origianl_hit_iter->start_time_sec));
            REQUIRE(deserialized_hit_iter->duration_sec == Catch::Approx(origianl_hit_iter->duration_sec));
            REQUIRE(deserialized_hit_iter->rate_index == origianl_hit_iter->rate_index);
            REQUIRE(deserialized_hit_iter->drift_rate_Hz_per_sec == Catch::Approx(origianl_hit_iter->drift_rate_Hz_per_sec));
            REQUIRE(deserialized_hit_iter->power == Catch::Approx(origianl_hit_iter->power));
            REQUIRE(deserialized_hit_iter->time_span_steps == origianl_hit_iter->time_span_steps);
            REQUIRE(deserialized_hit_iter->snr == Catch::Approx(origianl_hit_iter->snr));
            REQUIRE(deserialized_hit_iter->bandwidth == Catch::Approx(origianl_hit_iter->bandwidth));
            REQUIRE(deserialized_hit_iter->binwidth == origianl_hit_iter->binwidth);

        } while(++deserialized_hit_iter != deserialized_hits.end() && ++origianl_hit_iter != original_hits.end());
    }
}


TEST_CASE("capnp scan hits", "[serialization]") {
    SECTION("scan with hits", "round trip a scan that has hits in two sequential coarse channels") {

        auto coarse_channels = std::map<int, std::shared_ptr<bliss::coarse_channel>>();
        auto first_cc = std::make_shared<bliss::coarse_channel>(1000 /* fch1 */,
                                                1 /* foff */,
                                                0 /* machine_id */,
                                                32 /* nbits */,
                                                1000000 /* nchans */,
                                                18 /* nsteps */,
                                                1 /* nifs */,
                                                "test" /* source_name */,
                                                4 /* src_dej */,
                                                2 /* src_raj */,
                                                6 /* telescope_id */,
                                                17 /* tsamp */,
                                                54321 /* tstart */,
                                                1 /* datatype */,
                                                0 /* az_start */,
                                                0 /* za_start */);
        std::list<bliss::hit> first_cc_hits;
        first_cc_hits.push_back({
            .start_freq_index = 10,
            .start_freq_MHz = 1010.0F,
            .start_time_sec = 4.2,
            .duration_sec = 17,
            .rate_index = 5,
            .drift_rate_Hz_per_sec = 2.1,
            .power = 10000,
            .time_span_steps = 16,
            .snr = 10.0,
            .bandwidth = 3,
            .binwidth = 1
        });
        first_cc->add_hits(first_cc_hits);

        auto second_cc = std::make_shared<bliss::coarse_channel>(1001000 /* fch1 */,
                                                1 /* foff */,
                                                0 /* machine_id */,
                                                32 /* nbits */,
                                                1000000 /* nchans */,
                                                18 /* nsteps */,
                                                1 /* nifs */,
                                                "test" /* source_name */,
                                                4 /* src_dej */,
                                                2 /* src_raj */,
                                                6 /* telescope_id */,
                                                17 /* tsamp */,
                                                54321 /* tstart */,
                                                1 /* datatype */,
                                                0 /* az_start */,
                                                0 /* za_start */);
        std::list<bliss::hit> hits_for_2nd_cc;
        hits_for_2nd_cc.push_back({
            .start_freq_index = 200,
            .start_freq_MHz = 1001200.0F,
            .start_time_sec = 4.2,
            .duration_sec = 17,
            .rate_index = 13,
            .drift_rate_Hz_per_sec = 3.9,
            .power = 10000,
            .time_span_steps = 16,
            .snr = 10.0,
            .bandwidth = 3,
            .binwidth = 1
        });
        second_cc->add_hits(hits_for_2nd_cc);

        coarse_channels.insert({0, first_cc});
        coarse_channels.insert({1, second_cc});
        auto original_scan = bliss::scan(coarse_channels);
        
        auto tempfname = std::tmpnam(nullptr);

        bliss::write_scan_hits_to_capnp_file(original_scan, tempfname);

        // Targets serialized to disk -- Read back and check round-trip

        auto deserialized_scan = bliss::read_scan_hits_from_capnp_file(tempfname);

        REQUIRE(deserialized_scan.fch1() == original_scan.fch1());
        REQUIRE(deserialized_scan.foff() == original_scan.foff());
        REQUIRE(deserialized_scan.nchans() == original_scan.nchans());
        REQUIRE(deserialized_scan.tstart() == original_scan.tstart());
        REQUIRE(deserialized_scan.tsamp() == original_scan.tsamp());
        REQUIRE(deserialized_scan.ntsteps() == original_scan.ntsteps());
        REQUIRE(deserialized_scan.source_name() == original_scan.source_name());

        auto deserialized_hits = deserialized_scan.hits();

        REQUIRE(deserialized_hits.size() == original_scan.hits().size());

        auto deserialized_hit_iter = deserialized_hits.begin();
        auto origianl_hit_iter = first_cc_hits.begin();
        do {
            REQUIRE(deserialized_hit_iter->start_freq_index == origianl_hit_iter->start_freq_index);
            REQUIRE(deserialized_hit_iter->start_freq_MHz == Catch::Approx(origianl_hit_iter->start_freq_MHz));
            REQUIRE(deserialized_hit_iter->start_time_sec == Catch::Approx(origianl_hit_iter->start_time_sec));
            REQUIRE(deserialized_hit_iter->duration_sec == Catch::Approx(origianl_hit_iter->duration_sec));
            REQUIRE(deserialized_hit_iter->rate_index == origianl_hit_iter->rate_index);
            REQUIRE(deserialized_hit_iter->drift_rate_Hz_per_sec == Catch::Approx(origianl_hit_iter->drift_rate_Hz_per_sec));
            REQUIRE(deserialized_hit_iter->power == Catch::Approx(origianl_hit_iter->power));
            REQUIRE(deserialized_hit_iter->time_span_steps == origianl_hit_iter->time_span_steps);
            REQUIRE(deserialized_hit_iter->snr == Catch::Approx(origianl_hit_iter->snr));
            REQUIRE(deserialized_hit_iter->bandwidth == Catch::Approx(origianl_hit_iter->bandwidth));
            REQUIRE(deserialized_hit_iter->binwidth == origianl_hit_iter->binwidth);

        } while(++deserialized_hit_iter != deserialized_hits.end() && ++origianl_hit_iter != first_cc_hits.end());
    }
}

