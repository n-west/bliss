
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <core/hit.hpp>
#include <file_types/hits_file.hpp>

#include <fmt/core.h>
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

        bliss::write_hits_to_file(original_hits, tempfname);

        auto deserialized_hits = bliss::read_hits_from_file(tempfname);

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
    SECTION("scan with hits", "round trip a scan that has hits") {

        bliss::scan test_scan;

        // test_scan.

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

        bliss::write_hits_to_file(original_hits, tempfname);

        auto deserialized_hits = bliss::read_hits_from_file(tempfname);

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

