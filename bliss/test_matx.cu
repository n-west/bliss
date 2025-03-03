
#include <iostream>

#include <matx.h>

int main(int argc, char **argv) {
    auto a = matx::make_tensor<float>({10});
    a.SetVals({1,2,3,4,5,6,7,8,9,10});

    printf("You should see the values 1-10 printed\n");
    matx::print(a);

#if MATX_ENABLE_VIZ
    // If you enabled visualization support, uncomment this line and
    // you should see a test.html file in this directory with a line plot
    matx::viz::line(a, "Sample Line", "X values", "Y values", "test.html");
#endif    
}