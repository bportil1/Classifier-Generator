namespace Classifier_Generator;

class classifier_generator_interfaace
{
    static void Main (string[] args)
    {
        Console.WriteLine("Prrrr?");
        var name = Console.ReadLine();
        var currentDate = DateTime.Now;
        Console.WriteLine($"{Environment.NewLine}Hello, {name}, on {currentDate:d} at {currentDate:t}!");
        console.Write($"{Environment.NewLine}Press any key...");
        Console.ReadKey(true);
    }

}
