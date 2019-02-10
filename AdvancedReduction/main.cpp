#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cassert>


constexpr char* kCacheName = "randcache.bin";
constexpr int64_t kNumbers = 9ll * 1024ll * 1024ll * 1024ll / 4ll;
//constexpr int64_t kNumbers = 1024ll * 1024ll * 1024ll;

std::vector<float> GenerateAndWrite()
{
	std::vector<float> result;
	result.reserve(kNumbers);

	srand(42);
	for (int64_t i = 0; i < kNumbers; ++i)
		result.push_back(float(rand() % 11 - 5));

	auto file = fopen(kCacheName, "wb");
	assert(file != nullptr);
	auto written = fwrite(result.data(), sizeof(float), kNumbers, file);
	assert(written == kNumbers);
	fclose(file);

	return result;
}

size_t NumbersInCache()
{
	auto file = fopen(kCacheName, "rb");
	if (file == nullptr)
		return 0;

	fseek(file, 0, SEEK_END);
	auto pos = _ftelli64(file);
	fclose(file);

	return pos / sizeof(float);
}

std::vector<float> ReadNumbers()
{
	std::vector<float> result(kNumbers);

	auto file = fopen(kCacheName, "rb");
	auto red = fread(result.data(), sizeof(float), kNumbers, file);
	assert(red == kNumbers);
	fclose(file);

	return result;
}

std::vector<float> GetNumbers()
{
	if (NumbersInCache() < kNumbers)
		return GenerateAndWrite();

	return ReadNumbers();
}

void cudaReduce(const std::vector<float>& data);

int main(int argc, char** argv)
{
	std::chrono::high_resolution_clock::time_point begin;
	std::chrono::high_resolution_clock::time_point end;
	float controlResult = 0;

	std::vector<float> hostNumbers = GetNumbers();
	
	printf("Serial CPU started...\r\n");
	begin = std::chrono::high_resolution_clock::now();
	controlResult = 0;
#pragma loop(no_parallel)
	for (int64_t i = 0; i < kNumbers; ++i)
		controlResult += hostNumbers[i];
	end = std::chrono::high_resolution_clock::now();


	printf("Result: %.3f\r\n", controlResult);
	printf("Elapsed time on CPU (serial): %.3f ms\r\n", float(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) * 1e-6f);


	printf("Parallel CPU started...\r\n");
	begin = std::chrono::high_resolution_clock::now();
	controlResult = 0;
	auto data = hostNumbers.data();
#pragma omp parallel for reduction(+:controlResult)
	for (int64_t i = 0; i < kNumbers; ++i)
		controlResult += data[i];
	end = std::chrono::high_resolution_clock::now();


	printf("Result: %.3f\r\n", controlResult);
	printf("Elapsed time on CPU (parallel): %.3f ms\r\n", float(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) * 1e-6f);


	cudaReduce(hostNumbers);

	system("pause");

	return 0;
}