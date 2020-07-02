#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <iostream> 
#include <curand_kernel.h>

const int S = 500; //Количество точек, в которых будет расчитаноо значение функции на первом шаге итерации
const int n = 10; //Количество лучших участков
const int m = 5; //Количество выбранных участков. Участки с лучшими значениями функции после значений на лучших участках
const int N = 50; //Количество значений, вычисляемых в окрестностях лучших участков
const int M = 50; //Количество значений, вычисляемых в окрестностях выбраных участков
const int P = 50; //Количество пробных вокруг лучших, при выборе потенцильных точек
const float d = 0.5; //Размер окрестности 

const float MIN = -10;
const float MAX = 10;
const int iterations = 2;
const int numb_coordinates = 7;
const int vector_size = numb_coordinates + 1;

const int threadsPerBlock = 256;

void print_matrix(float * matrix, int str);
void copy_min_values(float * source, float * dest, int numb_source_p, int numb_dest_p);
int get_numb_stat();
void reading_data(float * data);
int check_test(float * test, float * best);

__device__ __host__ float function_for_calculation(float * training, float * cord) {
    float ret = 0;
    float result = 0;
    for(int i = 0; i < 87500; i++) {
        for(int j = 0; j < vector_size - 1; j++) {
            result += training[j + i * vector_size] * cord[j]; //Скалярное произведение
        }
        result *= -training[vector_size - 1 + i * vector_size]; //-y * <x,w>
        result = exp(result);
        result = log(1 + result);
        ret += result;
    }
    return ret;
}

__global__ void setup_kernel (curandState * state, unsigned long seed ) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init (seed, id,0, &state[id]);
}
 
__global__ void fill_rand(curandState * globalState, float * vec, float * training, int size) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;    
    if(ind <= size && ind >= 0  && (ind % vector_size != vector_size - 1)) {
        curandState localState = globalState[ind];
        float RANDOM = curand_uniform(&localState);
        globalState[ind]= localState;
        __syncthreads();
        vec[ind] = MIN + RANDOM * (MAX - MIN);
        __syncthreads();
    } 
    else if(ind % vector_size == vector_size - 1){
        vec[ind] = function_for_calculation(training, vec + ind - vector_size + 1);
        __syncthreads();
    }
}

__global__ void local_rand(curandState * globalState, float * vec, float * local, float * training, int size, int numb_rand_p) {
    int ind = threadIdx.x + blockIdx.x * blockDim.x;    
    if(ind < size && ind >= 0 && (ind % vector_size != vector_size - 1)) {
        int i = ind / (numb_rand_p * vector_size);
        int j = ind % vector_size;
        int index_point = j + i * vector_size;

        float start = local[index_point] - d / 2;
        float end = local[index_point] + d / 2;    
        if(MAX < end) 
            end = MAX;
        if(MIN > start) 
           end = MIN;

        curandState localState = globalState[ind];
        float RANDOM = curand_uniform(&localState);
        float result = start + RANDOM * (end - start);
        globalState[ind]= localState;
        __syncthreads();
        vec[ind]= result;
    } else if(ind % vector_size == vector_size - 1){
        vec[ind] = function_for_calculation(training, vec + ind - vector_size + 1);
    }
}

void calculation_function_values(float * random_points, curandState * device_states, float * device_training) {
    int size = vector_size * S;
    float * device_vector;
    cudaMalloc(&device_vector, size * sizeof(float));

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    fill_rand <<<blocksPerGrid, threadsPerBlock>>>(device_states, device_vector, device_training, size);
    cudaDeviceSynchronize();

    cudaMemcpy(random_points, device_vector, sizeof(float) * size, cudaMemcpyDeviceToHost); 
    cudaFree(device_vector);
}

void point_initialization(float * source_point, float * new_points, float * device_training, int numb_start_p, int numb_rand_p, int numb_get_p, curandState * device_states) {
    int size = vector_size * numb_start_p * numb_rand_p;
    float * device_vector, * device_local;
    cudaMalloc(&device_vector, size * sizeof(float));
    cudaMalloc(&device_local, vector_size * numb_start_p * sizeof(float));
    cudaMemcpy(device_local, source_point, sizeof(float) * vector_size * numb_start_p, cudaMemcpyHostToDevice);

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    local_rand <<<blocksPerGrid, threadsPerBlock>>>(device_states, device_vector, device_local, device_training, size, numb_rand_p);
    cudaDeviceSynchronize();

    if(numb_start_p * numb_rand_p == numb_get_p) {
        cudaMemcpy(new_points, device_vector, sizeof(float) * size, cudaMemcpyDeviceToHost); 
    } else {
        float * buffer = (float *)malloc(sizeof(float) * size);
        cudaMemcpy(buffer, device_vector, sizeof(float) * size, cudaMemcpyDeviceToHost); 
        copy_min_values(buffer, new_points, numb_start_p * numb_rand_p, numb_get_p);
        free(buffer);
    }
    cudaFree(device_vector);
    cudaFree(device_local);
}

void merger(float * buf, float * random_points, float * points_in_loc_best, float * points_in_loc_potential) {
    memcpy(buf, random_points, sizeof(float) * S * vector_size);    
    memcpy(buf + S * vector_size, points_in_loc_best, sizeof(float) * N * n * vector_size);    
    memcpy(buf + S * vector_size + N * n * vector_size, points_in_loc_potential, sizeof(float) * M * m * vector_size);    
}

void artificial_bee_colony_method(float * random_points, float * best_points, float * potential_points, float * points_in_loc_best, float * points_in_loc_potential, 
                                                                                float * buf, curandState * device_states, float * device_training, int numb_points_buf) {
    calculation_function_values(random_points, device_states, device_training);
    copy_min_values(random_points, best_points, S, n);
    point_initialization(best_points, potential_points, device_training, n, P, m, device_states);
    point_initialization(best_points, points_in_loc_best, device_training, n, N, n * N, device_states);
    point_initialization(potential_points, points_in_loc_potential, device_training, m, M, m * M, device_states);

    merger(buf, random_points, points_in_loc_best, points_in_loc_potential);

    for(int i = 0; i < iterations; i++) {
        calculation_function_values(random_points, device_states, device_training);
        merger(buf, random_points, points_in_loc_best, points_in_loc_potential);
        copy_min_values(buf, best_points, numb_points_buf, n);
        point_initialization(best_points, potential_points, device_training, n, P, m, device_states);
        point_initialization(best_points, points_in_loc_best, device_training, n, N, N * n, device_states);
        point_initialization(potential_points, points_in_loc_potential, device_training, m, M, M * m, device_states);
    }

    // print_matrix(random_points, S);
    // print_matrix(best_points, n);
    // print_matrix(potential_points, m);
    // print_matrix(points_in_loc_best, N * n);
    // print_matrix(points_in_loc_potential, M * m);
    // print_matrix(buf, numb_points_buf);
}

int main() {
    double t, t2;    
    float * data = (float *)malloc(sizeof(float) * 100000 * 8);
    reading_data(data);
    float * test = (float *)malloc(sizeof(float) * 12500 * vector_size);
    float * device_training;
    cudaMalloc(&device_training, 87500 * vector_size * sizeof(float));
    
    int numb_state = get_numb_stat(), blocksPerGrid = (numb_state + threadsPerBlock - 1) / threadsPerBlock, numb_points_buf = S + N * n + M * m;
    curandState * device_states;
    cudaMalloc(&device_states, numb_state * sizeof(curandState));
    setup_kernel <<<blocksPerGrid, threadsPerBlock>>> ( device_states, time(NULL));

    float * random_points = (float *) malloc(sizeof(float) * vector_size * S);
    float * best_points = (float *) malloc(sizeof(float) * vector_size * n); //Для n точек
    float * potential_points = (float *) malloc(sizeof(float) * vector_size * m); //Для m точек
    float * points_in_loc_best = (float *) malloc(sizeof(float) * vector_size * N * n); //Для N точек
    float * points_in_loc_potential = (float *) malloc(sizeof(float) * vector_size * M * m); //Для M точек
    float * buf = (float *) malloc(sizeof(float) * vector_size * numb_points_buf);

    int number_experiment = 1, start_test_area;
    float percent_correctly = 0, number_correct_exp = 0;
    for(int i = 0; i < number_experiment; i++) {
        start_test_area = rand() % (87500);
        memcpy(test, data + start_test_area * vector_size, sizeof(float) * 12500 * vector_size);
        cudaMemcpy(device_training, data, sizeof(float) * start_test_area * vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(device_training + start_test_area * vector_size, data + (start_test_area + 12500) * vector_size, sizeof(float) * (87500 - start_test_area ) * vector_size, cudaMemcpyHostToDevice);

        t = clock();
        artificial_bee_colony_method(random_points, best_points, potential_points, points_in_loc_best, points_in_loc_potential, buf, device_states, device_training, numb_points_buf);
        number_correct_exp = check_test(test, best_points);
        t2 = clock();
        printf("Experiment #%d\n", i + 1);
        // print_matrix(best_points, 1);
        printf("Elapsed time (sec.): %.6f\n", difftime(t2, t)/1000000);
        printf("Number of correctly classified objects: %d\n\n", (int)number_correct_exp);
        percent_correctly += number_correct_exp / 12500;
    }
    printf("Average percentage of correctly classified objects: %.2f%%\n", percent_correctly / number_experiment * 100);

    free(random_points);
    free(best_points);
    free(potential_points);
    free(points_in_loc_best);
    free(points_in_loc_potential);
    free(buf);
    free(data);

    cudaFree(device_states);

    return 0;
}

int check_test(float * test, float * best) {
    int count = 0;
    float result = 0;
    for(int i = 0; i < 12500; i++) {
        for(int j = 0; j < 7; j++) {
            result += test[j + i * vector_size] * best[j]; //Скалярное произведение
        }
        // printf("%f %f \n", result, test[vector_size - 1 + i * vector_size]);
        if(result < 0) {
            result = -1;
        }
        else {
            result = 1;
        }
        if(test[vector_size - 1 + i * vector_size] == result)
            count++;
        result = 0;
    }
    // printf("%d\n", count);   
    return count; 
}

void reading_data(float * data) {
    srand(time(NULL));
    FILE *mf = fopen ("data2.txt","r");
    if (mf == NULL) {
        printf("Ошибка открытия файла\n");
        return;
    }
    char str[100], *estr, *istr;
    int item_counter = 0, line_counter = 0;
    while (1)
    {
        estr = fgets (str,sizeof(str),mf);
        if (estr == NULL) {
            if ( feof (mf) != 0) {  
                break;
            }
            else {
                break;
            }
        }
        item_counter = 0;
        istr = strtok(str, " ");
        while(istr != NULL) {
            float var;
            sscanf(istr, "%f", &var);
            data[item_counter + line_counter * vector_size] = var;
            istr = strtok(NULL, " ");
            item_counter++;
        }
        line_counter++;
    }
    fclose (mf);    
}

int get_numb_stat() {
    int max = vector_size * S;
    if(max < vector_size * N * n) 
        max = vector_size * N * n;
    if(max < vector_size * M * m)
        max = vector_size * M * m;
    return max;
}

void copy_min_values(float * source, float * dest, int numb_source_p, int numb_dest_p) {
    int min; 
    int count = 0;
    for (int i = 0; i < numb_dest_p; i ++) {
        min = (vector_size - 1);
        for (int j = 0; j < numb_source_p; j ++) {
            if (source[(vector_size - 1) + j * vector_size] < source[min])
                min = (vector_size - 1) + j * vector_size;
        }
        memcpy(dest + count, source + min - vector_size + 1, sizeof(float) * vector_size);
        count += vector_size;
        source[min] = 9999999.9;
    }
}

void print_matrix(float * matrix, int str){
    for (int i = 0; i < str; i++)
    {
        for(int j = 0; j < vector_size; j++) {
            printf("%f ", matrix[j + i * vector_size]);
        }
        printf("\n");
    }
    printf("\n");
}