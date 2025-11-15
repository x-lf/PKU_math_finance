"""
币安期货数据获取工具
支持U本位期货和币本位期货数据获取

币本位期货API限制：
- startTime 与 endTime 之间最多只可以相差200天
- 默认返回 startTime 与 endTime 之间最接近 endTime的 limit 条数据
- startTime, endTime 均未提供的, 将会使用当前时间为 endTime, 200天前为 startTime
- 仅提供 startTime 的, 将会使用 startTime 之后200天作为默认 endTime (至多为当前时间)
- 仅提供 endTime 的, 将会使用endTime 之前200天作为默认 startTime

使用方法：
1. 设置symbol、时间范围、间隔等参数
2. 选择api_type: 'usdt' for U本位期货, 'coin' for 币本位期货
3. 运行程序获取数据
"""

import aiohttp
from itertools import chain
import asyncio
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Binance API URLs
BINANCE_USDT_API_URL = "https://fapi.binance.com/fapi/v1/klines"    # U本位
BINANCE_COIN_API_URL = "https://dapi.binance.com/dapi/v1/klines"    # 币本位

# 数据保存配置
DATA_DIR = "data"

# 代理配置
PROXY_CONFIG = {
    "http": "http://127.0.0.1:7890"
}

# 频率限制配置
RATE_LIMIT_CONFIG = {
    "max_weight_per_minute": 2400,  # 币安期货API每分钟最大权重
    "weight_per_request": 10,       # klines接口使用limit=1500时的权重
    "safety_factor": 0.8,           # 安全系数，使用80%的限制
    "requests_per_batch": 160,      # 每批次请求数量 (240*10=2400权重)
    "delay_between_batches": 120,    # 批次间延迟（秒）
    "request_delay": 0.25,          # 单个请求间延迟（秒）
    "retry_delay": 5,               # 遇到限制时的重试延迟（秒）
    "max_retries": 3                # 最大重试次数
}


def ensure_data_directory():
    """确保data目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logging.info(f"Created data directory: {DATA_DIR}")


def get_filename(symbol, start_year, end_year, interval):
    """生成文件名"""
    return f"{symbol}_{interval}_{start_year}_{end_year}.csv"


def get_existing_data_range(base_filepath, save_format='csv'):
    """获取已存在文件的数据时间范围，支持多种格式"""
    # 根据格式确定实际文件路径
    format_extensions = {'csv': '.csv', 'pickle': '.pkl', 'txt': '.txt'}
    base_name = base_filepath.rsplit('.', 1)[0]
    filepath = base_name + format_extensions[save_format]

    if not os.path.exists(filepath):
        return None, None

    try:
        # 根据格式读取文件
        if save_format == 'csv':
            df = pd.read_csv(filepath)
        elif save_format == 'pickle':
            df = pd.read_pickle(filepath)
        elif save_format == 'txt':
            df = pd.read_csv(filepath, sep='\t')
        else:
            raise ValueError(f"不支持的文件格式: {save_format}")

        if len(df) == 0:
            return None, None

        # 获取最后一条记录的时间（open_time列，原始时间戳）
        last_time_ms = df['open_time'].iloc[-1]
        last_time = datetime.fromtimestamp(last_time_ms / 1000)

        first_time_ms = df['open_time'].iloc[0]
        first_time = datetime.fromtimestamp(first_time_ms / 1000)

        return first_time, last_time
    except Exception as e:
        logging.warning(f"读取已有数据失败: {e}")
        return None, None


async def fetch_klines(session, symbol, start_time, end_time, interval, api_type='usdt'):
    """
    获取K线数据
    api_type: 'usdt' for U本位期货, 'coin' for 币本位期货
    """
    # 选择API URL
    api_url = BINANCE_USDT_API_URL if api_type == 'usdt' else BINANCE_COIN_API_URL

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1500
    }

    # 调试信息（如需要可以取消注释）
    # start_dt = datetime.fromtimestamp(start_time / 1000)
    # end_dt = datetime.fromtimestamp(end_time / 1000)

    for attempt in range(RATE_LIMIT_CONFIG["max_retries"] + 1):
        try:
            # 添加请求间延迟
            await asyncio.sleep(RATE_LIMIT_CONFIG["request_delay"])

            async with session.get(api_url, params=params, proxy=PROXY_CONFIG["http"]) as response:
                # 检查是否遇到频率限制
                if response.status == 429:
                    retry_after = int(response.headers.get(
                        'Retry-After', RATE_LIMIT_CONFIG["retry_delay"]))
                    logging.warning(f"触发频率限制，等待{retry_after}秒后重试")
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = await response.json()

                # 检查是否有错误响应
                if isinstance(data, dict) and 'code' in data:
                    if data.get('code') == -1121:  # Invalid symbol
                        logging.error(f"Invalid symbol or interval: {data}")
                        return []
                    elif data.get('code') == -1003:  # Too many requests
                        logging.warning(f"请求过多，重试中...")
                        await asyncio.sleep(RATE_LIMIT_CONFIG["retry_delay"])
                        continue
                    else:
                        logging.error(f"API Error: {data}")
                        return []

                # 成功获取数据
                return data

        except aiohttp.ClientError as e:
            if attempt < RATE_LIMIT_CONFIG["max_retries"]:
                await asyncio.sleep(RATE_LIMIT_CONFIG["retry_delay"])
            else:
                logging.error(f"请求失败，已重试{RATE_LIMIT_CONFIG['max_retries']}次")
                return []
        except Exception as e:
            logging.error(f"Unexpected error in fetch_klines: {e}")
            return []

    return []


def validate_time_range(start_time_ms, end_time_ms, api_type='usdt'):
    """
    验证时间范围是否符合币安API限制
    币本位API: startTime 与 endTime 之间最多只可以相差200天
    U本位API: 无此限制
    """
    if api_type == 'coin' and start_time_ms and end_time_ms:
        time_diff_days = (end_time_ms - start_time_ms) / (1000 * 60 * 60 * 24)
        if time_diff_days > 200:
            raise ValueError(f"币本位API时间范围不能超过200天，当前范围: {time_diff_days:.1f}天")
    return True


def adjust_time_range_for_coin_api(start_time_ms=None, end_time_ms=None):
    """
    根据币安币本位API规则调整时间范围
    - startTime, endTime 均未提供的, 将会使用当前时间为 endTime, 200天前为 startTime
    - 仅提供 startTime 的, 将会使用 startTime 之后200天作为默认 endTime (至多为当前时间)
    - 仅提供 endTime 的, 将会使用endTime 之前200天作为默认 startTime
    """
    current_time_ms = int(datetime.now().timestamp() * 1000)
    days_200_ms = 200 * 24 * 60 * 60 * 1000  # 200天的毫秒数

    if start_time_ms is None and end_time_ms is None:
        # 均未提供：使用当前时间为endTime，200天前为startTime
        end_time_ms = current_time_ms
        start_time_ms = end_time_ms - days_200_ms
    elif start_time_ms is not None and end_time_ms is None:
        # 仅提供startTime：使用startTime之后200天作为endTime（至多为当前时间）
        end_time_ms = min(start_time_ms + days_200_ms, current_time_ms)
    elif start_time_ms is None and end_time_ms is not None:
        # 仅提供endTime：使用endTime之前200天作为startTime
        start_time_ms = end_time_ms - days_200_ms

    # 验证时间范围
    validate_time_range(start_time_ms, end_time_ms, 'coin')

    return start_time_ms, end_time_ms


def _save_dataframe(df, filepath, save_format):
    """根据格式保存DataFrame的辅助函数"""
    if save_format == 'csv':
        df.to_csv(filepath, index=False)
    elif save_format == 'pickle':
        df.to_pickle(filepath)
    elif save_format == 'txt':
        df.to_csv(filepath, sep='\t', index=False)
    else:
        raise ValueError(f"不支持的保存格式: {save_format}")


def save_klines_data(filename, klines, append_mode=False, save_format='csv'):
    """将K线数据保存到文件，支持多种格式和追加模式"""
    try:
        # 只保留前11个字段
        klines_data = [kline[:11] for kline in klines]

        # 定义列名
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'turnover', 'trade_count', 'taker_buy_volume',
            'taker_buy_turnover'
        ]

        # 创建DataFrame
        df = pd.DataFrame(klines_data, columns=columns)

        # 保存原始数据，不做任何时间转换
        # 保持原始时间戳格式（毫秒）

        # 根据保存格式确定文件扩展名
        format_extensions = {'csv': '.csv', 'pickle': '.pkl', 'txt': '.txt'}
        base_name = filename.rsplit('.', 1)[0]  # 移除原有扩展名
        filepath = os.path.join(DATA_DIR, base_name +
                                format_extensions[save_format])

        # 按open_time排序以确保数据顺序正确
        df = df.sort_values('open_time')

        # 检查重复数据（基于open_time）
        duplicates = df.duplicated(subset=['open_time']).sum()
        if duplicates > 0:
            logging.warning(f"发现{duplicates}条重复数据，已移除")
            df = df.drop_duplicates(subset=['open_time'])

        # 根据模式保存数据
        if append_mode and os.path.exists(filepath):
            # 读取已有数据
            if save_format == 'csv':
                existing_df = pd.read_csv(filepath)
            elif save_format == 'pickle':
                existing_df = pd.read_pickle(filepath)
            elif save_format == 'txt':
                existing_df = pd.read_csv(filepath, sep='\t')

            # 合并数据
            combined_df = pd.concat([existing_df, df])
            # 按open_time排序并去重
            combined_df = combined_df.sort_values('open_time')
            combined_df = combined_df.drop_duplicates(subset=['open_time'])

            # 保存合并后的数据
            _save_dataframe(combined_df, filepath, save_format)
            logging.info(f"已追加 {len(klines_data)} 条数据，总计 {len(combined_df)} 条")
        else:
            # 正常保存（覆盖模式）
            _save_dataframe(df, filepath, save_format)
            logging.info(f"已保存 {len(klines_data)} 条数据")

    except Exception as e:
        logging.error(f"Failed to save klines to CSV: {e}")


async def main(symbol, start_year, end_year, interval, save_format='csv', api_type='usdt'):
    """
    主函数
    api_type: 'usdt' for U本位期货, 'coin' for 币本位期货
    """
    # 确保数据目录存在
    ensure_data_directory()

    # 生成文件名
    filename = get_filename(symbol, start_year, end_year, interval)

    api_name = "币本位" if api_type == 'coin' else "U本位"
    logging.info(
        f"获取 {symbol} {interval} {api_name}数据 ({start_year}-{end_year})")

    # 检查已有数据，实现增量更新
    filepath = os.path.join(DATA_DIR, filename)
    _, existing_end = get_existing_data_range(filepath, save_format)

    append_mode = False
    actual_start_time = None

    if existing_end:
        # 计算需要新增的时间范围
        interval_minutes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
                            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
                            '12h': 720, '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200}

        if interval in interval_minutes:
            next_start = existing_end + \
                timedelta(minutes=interval_minutes[interval])

            if next_start >= datetime(end_year + 1, 1, 1):
                logging.info("数据已是最新，无需更新")
                return

            # 启用增量模式
            actual_start_time = next_start
            append_mode = True
            logging.info(f"检测到已有数据到 {existing_end}，从 {next_start} 开始增量更新")
        else:
            logging.warning(f"不支持的时间间隔: {interval}，执行完整更新")
            os.remove(filepath)
    else:
        logging.info("未检测到已有数据，执行完整获取")

    # 创建连接器以避免Windows上的DNS问题
    connector = aiohttp.TCPConnector(use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        all_data = []  # 收集所有数据

        day_timestamps = get_year_timestamps(
            start_year, end_year, interval, actual_start_time, api_type)
        total_requests = len(day_timestamps)
        batch_size = RATE_LIMIT_CONFIG["requests_per_batch"]
        total_batches = (total_requests + batch_size - 1) // batch_size

        logging.info(
            f"开始获取数据: {total_requests}个时间段, {total_batches}批次, 预计{total_batches * RATE_LIMIT_CONFIG['delay_between_batches'] / 60:.1f}分钟")

        batch_count = 0
        for i in range(len(day_timestamps)):
            start_time = day_timestamps[i][0]
            end_time = day_timestamps[i][1]

            tasks.append(fetch_klines(session, symbol, start_time,
                         end_time, interval, api_type))

            # 使用配置的批次大小
            if (i + 1) % batch_size == 0 or i == len(day_timestamps) - 1:
                batch_count += 1
                logging.info(
                    f"批次 {batch_count}/{total_batches} ({len(tasks)}个请求)")

                results = await asyncio.gather(*tasks)
                batch_data = list(chain.from_iterable(results))
                all_data.extend(batch_data)

                logging.info(
                    f"完成 {batch_count}/{total_batches}, 获得{len(batch_data)}条数据, 总计{len(all_data)}条")

                tasks = []

                # 添加延迟以避免频率限制（除非是最后一批）
                if batch_count < total_batches:
                    logging.info(
                        f"等待{RATE_LIMIT_CONFIG['delay_between_batches']}秒...")
                    await asyncio.sleep(RATE_LIMIT_CONFIG["delay_between_batches"])

        # 一次性保存所有数据
        if all_data:
            logging.info(f"保存{len(all_data)}条数据到文件...")
            save_klines_data(filename, all_data, append_mode, save_format)
        else:
            logging.warning("未获取到数据!")

# 输入年份，返回该年份所有日期的开始和结束时间戳对的列表


def get_year_timestamps(start_year, end_year, interval, actual_start_time=None, api_type='usdt'):
    """
    生成时间戳列表
    api_type: 'usdt' for U本位期货, 'coin' for 币本位期货
    币本位API有200天限制，需要特殊处理
    """
    if actual_start_time:
        start = actual_start_time
    else:
        start = datetime(start_year, 1, 1)
    # 修复：确保包含end_year的完整年份数据，到下一年1月1日（不包含）
    end = datetime(end_year + 1, 1, 1)
    timestamps = []

    current_time = datetime.now()
    if end > current_time:
        end = current_time
        logging.info(f"结束时间调整为当前时间: {end}")

    # 币本位API特殊处理：将长时间范围分割成200天的块
    if api_type == 'coin':
        total_days = (end - start).days
        if total_days > 200:
            # logging.warning(f"币本位API限制：总时间范围{total_days}天超过200天限制")
            logging.info(f"将分批获取数据，每批最多200天")
            # 注意：这里不调整时间范围，而是在后续逻辑中分批处理

    """"if interval == '1m':
        timedelta_ = timedelta(days=1)
    elif interval == '1h':
        timedelta_ = timedelta(weeks=8)
    elif interval == '4h':
        timedelta_ = timedelta(weeks=35)
    elif interval == '1d':
        timedelta_ = timedelta(weeks=214)"""

    # 动态设置时间间隔的跨度，确保每次请求不超过1500条数据
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }

    if interval not in interval_minutes:
        raise ValueError(f"Unsupported interval: {interval}")

    # 计算每次请求的时间跨度（1500条数据）
    minutes_per_request = 1500 * interval_minutes[interval]
    timedelta_ = timedelta(minutes=minutes_per_request)

    # 币本位API需要考虑200天限制
    if api_type == 'coin':
        max_days = 199  # 使用199天确保不超过200天限制
        max_timedelta = timedelta(days=max_days)

        # 如果单次请求跨度超过199天，需要调整
        if timedelta_ > max_timedelta:
            timedelta_ = max_timedelta
            # logging.warning(f"币本位API限制：单次请求跨度从{minutes_per_request/60/24:.1f}天调整为{max_days}天")

    # 生成时间戳
    while start < end:
        next_time = start + timedelta_

        # 币本位API额外检查：确保单次请求不超过199天
        if api_type == 'coin':
            max_end_time = start + timedelta(days=199)
            if next_time > max_end_time:
                next_time = max_end_time

        # 确保不超过结束时间
        if next_time > end:
            next_time = end

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(next_time.timestamp() * 1000 - 1)  # 减1毫秒避免重叠

        # 验证时间范围
        time_diff_days = (end_ms - start_ms) / (1000 * 60 * 60 * 24)
        if time_diff_days <= 0:
            logging.warning(f"跳过无效时间范围: {time_diff_days:.1f}天")
            break

        if api_type == 'coin' and time_diff_days > 200:
            logging.warning(f"跳过超过200天限制的时间范围: {time_diff_days:.1f}天")
            start = next_time
            continue

        timestamps.append((start_ms, end_ms))

        start = next_time
        # 如果已经到达或超过结束时间，退出循环
        if start >= end:
            break

    return timestamps


async def fetch_coin_margined_data(symbol, start_year, end_year, interval='1h', save_format='csv'):
    """
    便捷函数：获取币本位期货数据

    Args:
        symbol: 币本位合约符号，如 'BTCUSD_PERP', 'ETHUSD_PERP'
        start_year: 开始年份
        end_year: 结束年份
        interval: 时间间隔，默认'1h'
        save_format: 保存格式，默认'csv'
    """
    logging.info(f"开始获取币本位期货数据: {symbol}")
    await main(symbol, start_year, end_year, interval, save_format, 'coin')


async def fetch_usdt_margined_data(symbol, start_year, end_year, interval='1h', save_format='csv'):
    """
    便捷函数：获取U本位期货数据

    Args:
        symbol: U本位合约符号，如 'BTCUSDT', 'ETHUSDT'
        start_year: 开始年份
        end_year: 结束年份
        interval: 时间间隔，默认'1h'
        save_format: 保存格式，默认'csv'
    """
    logging.info(f"开始获取U本位期货数据: {symbol}")
    await main(symbol, start_year, end_year, interval, save_format, 'usdt')

if __name__ == "__main__":
    # 在Windows上设置事件循环策略以避免aiodns问题
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # 配置参数
    symbol = "SOLUSDT"  # 币本位后要加_PERP
    start_year = 2020
    end_year = 2025
    interval = '15m'
    save_format = 'txt'  # 可选: 'csv', 'pickle', 'txt'
    api_type = 'usdt'  # 'usdt' for U本位期货, 'coin' for 币本位期货

    # 运行主程序
    asyncio.run(main(symbol, start_year, end_year,
                interval, save_format, api_type))

    # 示例：获取U本位数据
    """"
    symbol = "BTCUSDT"  # U本位合约符号
    start_year = 2023
    end_year = 2024
    interval = '15m'
    save_format = 'csv'
    api_type = 'usdt'
    asyncio.run(main(symbol, start_year, end_year, interval, save_format, api_type))
    """
